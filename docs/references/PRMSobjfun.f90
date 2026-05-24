module objfun_mod
  implicit none

  contains
    subroutine PRMSobjfun(hw_str, SCAcal, NRUNS, iSTEP, rnd_num, prmsOF)
      use ieee_arithmetic
      use byhw_constants, only: CAL_Q_FILE, HRU_PAREA_FILE, HRUID_FILE, OBJFUN_FILENAME, &
                                START_DATE_MODEL, END_DATE_MODEL, &
                                START_DATE_OBS, END_DATE_OBS, &
                                MEDIAN_Q, MIN_Q, MAX_Q, SIM_Q, &
                                YEAR, MONTH, DAY
      implicit none

      ! Calculates the objective function for STEPS 1 - 4
      !   iSTEP 1 == VOLUME
      !   iSTEP 2 == HIGH
      !   iSTEP 3 == LOW
      !   iSTEP 4 == ALL

      ! Subroutine arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: SCAcal
        !! Flag to enable/disable SCA calibration
      integer, intent(in) :: NRUNS
      integer, intent(in) :: iSTEP
      integer, intent(in) :: rnd_num
      real, intent(inout) :: prmsOF

      ! Parameters
      integer, parameter :: NMONTHS = 12
      integer, parameter :: NOF = 7
        !! Number of objective functions

      ! Local variables
      logical, save :: first_call = .true.

      integer :: cday
        !! Current day number
      integer :: chru
        !! Current HRU number
      integer :: cmonth
        !! Current month number
      integer, save :: funit
        !! File unit for writing objective function results
      integer :: i
      integer :: ii
      integer, save :: num_model_days
      integer, save :: num_obs_days
        !! Number of days of values
      integer, save :: nhru
        !! Number of HRUs in model
      integer, save :: num_months

      real :: NRMSE(NOF)
      real :: OFaet
      real :: OFrch
      real :: OFrun
      real :: OFsca
      real :: OFsom
      real :: weight(4, NOF)

      ! Allocatable arrays
      character(len=6), allocatable, save :: HRUid(:)

      integer, allocatable, save :: efc(:)
      integer, allocatable, save :: model_dates(:, :)
      integer, allocatable, save :: obs_dates(:, :)

      real, allocatable, save :: mnmth(:, :)
      real, allocatable, save :: mth(:, :)
        !! Monthly averages (4 x totalmonths), Qmed, Qmin, Qmax, sim_vals
      integer, allocatable, save :: nmnmth(:)
      integer, allocatable, save :: nmth(:)
        !! Number of days in each month
      real, allocatable, save :: pct_area(:)
      real, allocatable, save :: obs_vals(:, :)
      real, allocatable, save :: sim_vals(:)

      real, allocatable, save :: AET(:, :)
      real, allocatable, save :: RCH(:, :)
      real, allocatable, save :: RUN(:, :)
      real, allocatable, save :: SCA(:, :)
      real, allocatable, save :: SOM(:, :)

      ! =========================
      if (first_call) then
        ! The stuff we only run once
        first_call = .false.

        open (20, file=HRUID_FILE, status='old')
        read (20, *) nhru

        ! Allocate the some of the nhru dimensioned arrays
        allocate(HRUid(nhru))
        allocate(pct_area(nhru))

        do chru = 1, nhru
          read (20, *) HRUid(chru)
        end do
        close (20)

        open (20, file=HRU_PAREA_FILE, status='old')
        read (20, *)    ! nhru

        do chru = 1, nhru
          read (20, *) pct_area(chru)
        end do
        close (20)

        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ! Get the total number of output rows in the statvar file
        call get_model_info(hw_str, START_DATE_MODEL, END_DATE_MODEL, model_dates)

        num_model_days = size(model_dates, 2)

        ! Load the calibration observed streamflow
        call read_observed_streamflow(hw_str, START_DATE_OBS, END_DATE_OBS, num_months, obs_vals, efc, obs_dates)

        num_obs_days = size(obs_dates, 2)

        ! DEBUG:
        write(*, '(a, i0)') 'Number of obs days (num_obs_days): ', num_obs_days
        write(*, '(a, i0)') 'Number of model days (num_model_days): ', num_model_days

        ! Allocate arrays

        ! Simulated streamflow from statvar
        allocate(sim_vals(num_obs_days))

        allocate(mth(4, num_months))
        allocate(nmth(num_months))
        allocate(mnmth(4, NMONTHS))
        allocate(nmnmth(NMONTHS))

        allocate(AET(nhru, num_model_days))
        allocate(RCH(nhru, num_model_days))
        allocate(RUN(nhru, num_model_days))
        allocate(SCA(nhru, num_model_days))
        allocate(SOM(nhru, num_model_days))

        ! =========================
        !  OF1= mth with RANGE
        !  OF2= mnmth with RANGE
        !  OF3= mth with MEDIAN
        !  OF4= mnmth with MEDIAN
        !  OF5= daily with RANGE
        !  OF6= HIGH daily with MEDIAN
        !  OF7= LOW daily with MEDIAN

        ! Set the weights for the calibration steps
        weight = 1.0
        weight(1, 1:4) = 3.0  ! Step 1, weight volumes more
        weight(2, 6) = 3.0    ! Step 2 HIGH, weight HIGH flows most
        weight(3, 7) = 3.0    ! Step 3 LOW, weight LOW flows mosts
        weight(4, 5:7) = 3.0  ! Step 4 ALL, weight DAILY most

        ! Write out a header row for the console output for the calibration
        open(newunit=funit, file=OBJFUN_FILENAME//hw_str, status='replace')
        write(funit, '(a, a)') 'HW round step of_prms of_mth_range of_mnmth_range of_mth_median of_mnmth_median ', &
                               'of_daily_range of_high_daily_median of_low_daily_median of_run of_aet of_sca of_rch of_som'
      end if  ! first_call
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! --------------------------------------------------------------
      ! Model runs from 19801001 - 20100930 only calibrate streamflow on
      ! odd calendar years
      call read_simulated_streamflow(hw_str, obs_dates, sim_vals)

      ! Read the animation file
      call read_ani_file(hw_str, aet, rch, run, sca, som)

      ! Initialize arrays
      mth = 0.0
      nmth = 0
      mnmth = 0.0
      nmnmth = 0

      num_months = 0

      do cday = 1, num_obs_days
        ! Increment number of months on the first day of each month
        if (obs_dates(DAY, cday) == 1) num_months = num_months + 1

        cmonth = obs_dates(MONTH, cday)

        mnmth(MEDIAN_Q, cmonth) = mnmth(MEDIAN_Q, cmonth) + obs_vals(MEDIAN_Q, cday)
        mnmth(MIN_Q, cmonth) = mnmth(MIN_Q, cmonth) + obs_vals(MIN_Q, cday)
        mnmth(MAX_Q, cmonth) = mnmth(MAX_Q, cmonth) + obs_vals(MAX_Q, cday)
        mnmth(SIM_Q, cmonth) = mnmth(SIM_Q, cmonth) + sim_vals(cday)
        nmnmth(cmonth) = nmnmth(cmonth) + 1

        mth(MEDIAN_Q, num_months) = mth(MEDIAN_Q, num_months) + obs_vals(MEDIAN_Q, cday)
        mth(MIN_Q, num_months) = mth(MIN_Q, num_months) + obs_vals(MIN_Q, cday)
        mth(MAX_Q, num_months) = mth(MAX_Q, num_months) + obs_vals(MAX_Q, cday)
        mth(SIM_Q, num_months) = mth(SIM_Q, num_months) + sim_vals(cday)
        nmth(num_months) = nmth(num_months) + 1
      end do

      ! ----------------------------
      do ii = 1, 4
        mnmth(ii, :) = mnmth(ii, :) / nmnmth(:)
        mth(ii, :) = mth(ii, :) / nmth(:)
      end do

      ! NOTE: Uncomment if you want runoff output to a file
      ! close(75)
      ! open(75, file='roHW_'//hw_str, status='replace')
      ! write(75, '(a)') 'month mnmth_min mnmth_max mnmth_median mnmth_sim'

      ! do cmonth = 1, 12
      !   write(75, '(i2, 1x, 4(f12.5, 1x))') cmonth, mnmth(MIN_Q, cmonth), mnmth(MAX_Q, cmonth), &
      !                                        mnmth(MEDIAN_Q, cmonth), mnmth(SIM_Q, cmonth)
      ! end do

      ! ----------------------------
      !  OBS    --  4
      !  SIMmed --  1
      !  LOWER  --  2
      !  UPPER  --  3

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE mth with RANGE:
      NRMSE(1) = nrmse_range(mth(SIM_Q, :), mth(MAX_Q, :), mth(MIN_Q, :), num_months, weight(iSTEP, 1))

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE mnmth with RANGE:
      NRMSE(2) = nrmse_range(mnmth(SIM_Q, :), mnmth(MAX_Q, :), mnmth(MIN_Q, :), num_months, weight(iSTEP, 2))

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE mth with MEDIAN:
      NRMSE(3) = nmrse(mth(SIM_Q, :), mth(MEDIAN_Q, :), num_months, weight(iSTEP, 3))

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE mnmth with MEDIAN:
      NRMSE(4) = nmrse(mnmth(SIM_Q, :), mnmth(MEDIAN_Q, :), num_months, weight(iSTEP, 4))

      ! NRMSE daily with RANGE
      NRMSE(5) = nrmse_range(sim_vals, obs_vals(MAX_Q, :), obs_vals(MIN_Q, :), num_obs_days, weight(iSTEP, 5))

      !  NRMSE daily with EFCs
      !  1. large floods
      !  2. small floods
      !  3. high flow pulses
      !  4. low flows
      !  5. extreme low flows
      !  6. HIGH == 1,2,3
      !  7. LOW == 4,5

      ! HIGH EFCs (EFC <= 3)
      NRMSE(6) = nmrse_median_efc_high(sim_vals, obs_vals(MEDIAN_Q, :), efc, weight(iSTEP, 6))

      ! LOW EFCs (EFC > 3)
      NRMSE(7) = nmrse_median_efc_low(sim_vals, obs_vals(MEDIAN_Q, :), efc, weight(iSTEP, 7))

      prmsOF = sum(NRMSE)

      ! ==================================
      ! ==== Now look at the baselines:
      ! RUN
      call calcRUN(hw_str, model_dates, num_model_days, RUN, NRUNS, HRUid, pct_area, ofRUN)

      ! AET:
      call calcAET(model_dates, num_model_days, AET, NRUNS, HRUid, pct_area, ofAET)

      ! SCA
      if (SCAcal == 1) then
        call calcSCA(hw_str, model_dates, num_model_days, SCA, NRUNS, HRUid, pct_area, ofSCA)
      else
        ofSCA = 0.0
      end if

      ! RCH
      call calcRCH(model_dates, num_model_days, RCH, NRUNS, HRUid, pct_area, ofRCH)

      ! SOM
      call calcSOM(model_dates, num_model_days, SOM, NRUNS, HRUid, pct_area, ofSOM)

      ! WARNING: 2021-09-10 PAN - Should the denominator be adjusted
      !                           if SCA was NOT calibrated?
      prmsOF = prmsOF + (ofRUN + ofAET + ofSCA + ofRCH + ofSOM) / 5.0

      ! Write out the objective function values
      write(funit, 171) hw_str, rnd_num, iSTEP, prmsOF, (NRMSE(i), i=1, NOF), ofRUN, ofAET, ofSCA, ofRCH, ofSOM
      171 format(a, 1x, i0, 1x, i0, 1x, 13(f15.8, 1x))
    end
    ! ===================================================================


    subroutine get_model_info(hw_str, start_date, end_date, model_dates)
      use byhw_constants, only: YEAR, MONTH, DAY
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: start_date
      integer, intent(in) :: end_date
      integer, allocatable, intent(inout) :: model_dates(:, :)

      ! Local variables
      integer :: dy
      integer :: cdate
      integer :: funit
      integer :: ii
      integer :: ios
      integer :: junk_int
      integer :: mon
      integer :: num_days
      integer :: yr

      ! ---------------------------------------------------------------------
      open(newunit=funit, file='statvar_'//hw_str, iostat=ios, status='old')

      ! Get the daily simulated streamflow data:
      read(funit, *) junk_int
      do ii = 1, junk_int
        read(funit, *)
      end do

      num_days = 0

      do
        read(funit, *, iostat=ios) junk_int, yr, mon, dy
        if (ios /= 0) exit

        cdate = (yr * 10000) + (mon * 100) + dy
        if (cdate < start_date) cycle
        if (cdate > end_date) exit

        num_days = num_days + 1
      end do

      ! Holds year, month, day for each day in file
      allocate(model_dates(3, num_days))

      rewind(funit)

      ! Now read the dates in
      read(funit, *) junk_int
      do ii = 1, junk_int
        read(funit, *)
      end do

      ii = 1
      do
        if (ii > num_days) exit

        read(funit, *, iostat=ios) junk_int, yr, mon, dy
        if (ios /= 0) exit

        cdate = (yr * 10000) + (mon * 100) + dy
        if (cdate < start_date) cycle
        if (cdate > end_date) exit

        model_dates(YEAR, ii) = yr
        model_dates(MONTH, ii) = mon
        model_dates(DAY, ii) = dy
        ii = ii + 1
      end do

      close(funit)
    end subroutine

    subroutine read_ani_file(hw_str, aet, rch, run, sca, som)
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      real, intent(inout) :: aet(:, :)
      real, intent(inout) :: rch(:, :)
      real, intent(inout) :: run(:, :)
      real, intent(inout) :: sca(:, :)
      real, intent(inout) :: som(:, :)

      ! Local variables
      character(len=10) :: date
      character(len=10) :: header

      integer :: chru
      integer :: funit
      integer :: idx
      integer :: ios
      integer :: junk_int
      integer :: num_hru
      integer :: num_model_days

      ! ---------------------------------------------------------------------
      num_hru = size(aet, 1)
      num_model_days = size(aet, 2)

      ! Read the animation file
      open (newunit=funit, file='ani_'//hw_str//'.nhru', status='old')

      ! Read header line for animation file
      do
        read (funit, *) header
        if (header(1:3) == '10d') exit
      end do

      ! NOTE: PAN - how much of the 'base' data should be read in? Right now all of it is.
      do idx=1, num_model_days
        do chru = 1, num_hru
          read (funit, *, iostat=ios) date, junk_int, run(chru, idx), aet(chru, idx), rch(chru, idx), sca(chru, idx), som(chru, idx)
        end do
      end do

      close (funit)
    end subroutine

    subroutine read_observed_streamflow(hw_str, start_date, end_date, num_months, obs_vals, obs_efc, obs_dates)
      use byhw_constants, only: CAL_Q_FILE, MEDIAN_Q, MIN_Q, MAX_Q, YEAR, MONTH, DAY
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: start_date
      integer, intent(in) :: end_date

      integer, intent(inout) :: num_months
      real, allocatable, intent(inout) :: obs_vals(:, :)
      integer, allocatable, intent(inout) :: obs_efc(:)
      integer, allocatable, intent(inout) :: obs_dates(:, :)

      ! Local variables
      integer :: cdate
      integer :: cnt
      integer :: dy
      integer :: funit
      integer :: ios
      integer :: junk_int
      integer :: mon
      integer :: num_days
      integer :: yr

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Get the number of rows of calibration data
      open(newunit=funit, file=CAL_Q_FILE//hw_str, status='old')
      read(funit, *)  ! Skip header

      num_days = 0
      num_months = 0    ! Total number of months in date range

      ! First figure out how much data there is
      do
        read (funit, *, iostat=ios) junk_int, yr, mon, dy
        if (ios /= 0) exit

        cdate = (yr * 10000) + (mon * 100) + dy
        if (cdate < start_date) cycle
        if (cdate > end_date) exit

        if (mod(yr, 2) /= 0) then
          ! Only include odd years for calibration
          num_days = num_days + 1
          if (dy == 1) num_months = num_months + 1
        end if
      end do
      ! close(funit)

      allocate(obs_dates(3, num_days))
      allocate(obs_efc(num_days))
      allocate(obs_vals(3, num_days))

      rewind(funit)
      read(funit, *)  ! Skip header

      ! Read the observed statistical streamflow
      cnt = 1

      do
        if (cnt > num_days) exit
        ! read (funit, *, iostat=ios) junk_int, year, month, day, obs_efc(cnt), obs_median(cnt), obs_min(cnt), obs_max(cnt)
        read (funit, *, iostat=ios) junk_int, yr, mon, dy, obs_efc(cnt), obs_vals(MEDIAN_Q, cnt), &
                                    obs_vals(MIN_Q, cnt), obs_vals(MAX_Q, cnt)
        if (ios /= 0) exit

        cdate = (yr * 10000) + (mon * 100) + dy
        if (cdate < start_date) cycle
        if (cdate > end_date) exit

        if (mod(yr, 2) /= 0) then
          ! Only include odd years for calibration
          obs_dates(YEAR, cnt) = yr
          obs_dates(MONTH, cnt) = mon
          obs_dates(DAY, cnt) = dy
          cnt = cnt + 1
        end if
      end do

      close(funit)
    end subroutine

    subroutine read_simulated_streamflow(hw_str, obs_dates, sim_vals)
      use byhw_constants, only: YEAR, MONTH, DAY
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: obs_dates(:, :)
      real, intent(inout) :: sim_vals(:)

      ! Local variables
      integer :: cdate
      integer :: obs_cdate
      integer :: dy
      integer :: end_date
      integer :: funit
      integer :: ii
      integer :: ios
      integer :: junk_int
      integer :: mon
      integer :: num_days
      integer :: start_date
      integer :: yr

      ! ---------------------------------------------------------------------
      num_days = size(obs_dates, 2)
      start_date = (obs_dates(YEAR, 1) * 10000) + (obs_dates(MONTH, 1) * 100) + obs_dates(DAY, 1)
      end_date = (obs_dates(YEAR, num_days) * 10000) + (obs_dates(MONTH, num_days) * 100) + obs_dates(DAY, num_days)

      ! write(*, '(a, i0)') 'num_days: ', num_days
      ! write(*, '(a, i0)') 'start date: ', start_date
      ! write(*, '(a, i0)') 'end date: ', end_date

      open(newunit=funit, file='statvar_'//hw_str, iostat=ios, status='old')

      ! Get the daily simulated streamflow data:
      read(funit, *) junk_int
      do ii = 1, junk_int
        read(funit, *)
      end do

      ii = 1
      do
        if (ii > num_days) exit

        read(funit, *, iostat=ios) junk_int, yr, mon, dy, junk_int, junk_int, junk_int, sim_vals(ii)
        if (ios /= 0) exit

        cdate = (yr * 10000) + (mon * 100) + dy
        if (cdate < start_date) cycle
        if (cdate > end_date) exit

        if (mod(yr, 2) /= 0) then
          obs_cdate = (obs_dates(YEAR, ii) * 10000) + (obs_dates(MONTH, ii) * 100) + obs_dates(DAY, ii)

          if (cdate /= obs_cdate) then
            write(*, '(a, a, a, i0, 1x, i0, 1x, i0)') 'HW', hw_str, ' ERROR: Dates between obs and sim do not match: ', ii, &
                                                      obs_cdate, cdate
            STOP
          end if

          ii = ii + 1
        end if
      end do

      close(funit)
    end subroutine

    ! subroutine calcRUN(hw_str, iy, im, id, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
    subroutine calcRUN(hw_str, model_dates, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
      use byhw_constants, only: START_YM_RUN, END_YM_RUN, YEAR, MONTH, DAY
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: model_dates(:, :)
      integer, intent(in) :: nday
      real, intent(in) :: sim(:, :)
      integer, intent(in) :: NRUNS
      character(len=*), intent(in) :: HRUid(:)
      real, intent(in) :: pct_area(:)
      real, intent(inout) :: prmsOF

      ! Local variables
      logical, save :: first_call = .true.

      integer :: funit
      integer :: i
      integer :: ihru
      integer :: ios
        !! iostat results
      integer :: m
      integer :: model_cdate
      integer :: n
      integer :: ndate
      integer, save :: nhru
        !! Number of HRUs in the headwater
      integer :: nm
      integer, save :: nmths
      integer :: nn
      integer :: num_months
        !! Number of months within sim date range
      integer :: ny

      real :: q(3)

      ! Allocatable arrays
      integer, allocatable, save :: nmthsim(:)

      logical, allocatable, save :: use_hru(:)

      real, allocatable, save :: mth(:, :, :)
      real, allocatable, save :: mthsim(:, :)
      real, allocatable, save :: MWBMmax(:)
      real, allocatable, save :: MWBMmin(:)

      ! ---------------------------------------------------------------------
      ! --------------------------
      ! Normalizing all the monthly data using the min and max from MWBMq
      ! --------------------------
      ! beg_date = (model_dates(YEAR, 1) * 100) + model_dates(MONTH, 1)
      ! end_date = (model_dates(YEAR, nday) * 100) + model_dates(MONTH, nday)

      ! --------------------------
      ! --------------------------
      if (first_call) then
        first_call = .false.
        nhru = size(HRUid)

        ! Get MWBM outputs for HRU --> 3 values
        ! These have calculated error bounds (yr mth MWBMq min max)
        allocate(MWBMmax(nhru))
        allocate(MWBMmin(nhru))
        allocate(use_hru(nhru))

        ! Initially assume all HRUs will have valid data
        use_hru = .true.

        do ihru = 1, nhru
          open (newunit=funit, file='./RUN/HRU_'//HRUid(ihru), status='old')

          ! On the first HRU read through to compute the number of months
          if (ihru == 1) then
            nmths = 0
            do
              read(funit, *, iostat=ios) ny, nm
              if (ios /= 0) exit

              ndate = (ny * 100) + nm
              if (ndate < START_YM_RUN) cycle
              if (ndate > END_YM_RUN) exit

              nmths = nmths + 1
            end do

            ! Allocate the mth array to hold the runoff data
            allocate(mth(nhru, 3, nmths))

            rewind(funit)
          end if

          nmths = 0
          MWBMmin(ihru) = 0.0
          MWBMmax(ihru) = -99999.0

          do
            read(funit, *, iostat=ios) ny, nm, (q(i), i=1, 3)
            if (ios /= 0) exit

            ndate = (ny * 100) + nm
            if (ndate < START_YM_RUN) cycle
            if (ndate > END_YM_RUN) exit

            if (q(1) >= 0.0) then
              ! Only valid values are considered
              if (MWBMmin(ihru) > q(1)) MWBMmin(ihru) = q(1)
              if (MWBMmax(ihru) < q(1)) MWBMmax(ihru) = q(1)
            end if

            nmths = nmths + 1
            do i = 1, 3
              mth(ihru, i, nmths) = q(i)
            end do
          end do

          close(funit)

          if (MWBMmax(ihru) >= 0.0 .and. MWBMmin(ihru) >= 0.0) then
            do n = 1, nmths
              do i = 1, 3
                mth(ihru, i, n) = (mth(ihru, i, n) - MWBMmin(ihru)) / (MWBMmax(ihru) - MWBMmin(ihru))
              end do
            end do
          else
            use_hru(ihru) = .false.
            write(*, '(a, a, a, a)') 'HW', hw_str, ' WARNING: No runoff values for HRU ', trim(HRUid(ihru))
          end if
        end do

        if (.not. any(use_hru)) then
          ! All baseline data is missing/invalid
          write(*, '(a, a, a)') 'HW', hw_str, ' WARNING: No runoff baseline data for any HRUs'
        end if

        ! Allocate mthsim and nmthsim
        num_months = 0

        do n = 1, nday
            model_cdate = (model_dates(YEAR, n) * 100) + model_dates(MONTH, n)
            if (model_cdate < START_YM_RUN) cycle
            if (model_cdate > END_YM_RUN) exit

          ! NOTE: This will crash if the first day of the array is not one
          if (model_dates(DAY, n) == 1) num_months = num_months + 1
        end do

        allocate(mthsim(nhru,num_months))
        allocate(nmthsim(num_months))
      end if    ! first_call

      ! --------------------------
      ! --------------------------
      ! Init array
      mthsim = 0.0

      do ihru = 1, nhru
        if (use_hru(ihru)) then
          ! NOTE: PAN - populating nmthsim could be done just once when NRUNS=1
          ! Init array
          nmthsim = 0
          nn = 0

          ! Resample daily model output to normalized monthly means
          do n = 1, nday
            model_cdate = (model_dates(YEAR, n) * 100) + model_dates(MONTH, n)
            if (model_cdate < START_YM_RUN) cycle
            if (model_cdate > END_YM_RUN) exit

            ! NOTE: This will crash if the first day of the array is not one
            if (model_dates(DAY, n) == 1) nn = nn + 1

            ! m = model_dates(MONTH, n)

            mthsim(ihru, nn) = mthsim(ihru, nn) + sim(ihru, n)
            nmthsim(nn) = nmthsim(nn) + 1
          end do

          do n = 1, size(mthsim, 2)
            mthsim(ihru, n) = mthsim(ihru, n) / real(nmthsim(n))
            mthsim(ihru, n) = (mthsim(ihru, n) - MWBMmin(ihru)) / (MWBMmax(ihru) - MWBMmin(ihru))
          end do
        end if
      end do

      ! ----------------------------
      ! ----------------------------
      ! 1--OBS (1--MWBM, 2--lower, 3--upper)
      ! 2--SIM

      ! NRMSE mth:
      prmsof = 0.0

      do ihru = 1, nhru
        if (use_hru(ihru)) then
          prmsof = prmsof + nrmse_range(mthsim(ihru, :), mth(ihru, 3, :), mth(ihru, 2, :), nn, pct_area(ihru))
        end if
      end do

      ! -----------------------------------
      if (NRUNS == 0) then
        do ihru = 1, nhru
          if (use_hru(ihru)) then
            close (75)
            open (75, file='./RESULTS/'//'RUN_HRU'//HRUid(ihru), status='replace')
            write (75, '(a)') 'MONTH        MWBMq     MWBMlow    MWBMhigh        PRMS '

            do m = 1, nn
              write (75, 2) m, mth(ihru, 1, m), mth(ihru, 2, m), mth(ihru, 3, m), mthsim(ihru, m)
            end do
          end if
        end do
      end if

      2 format(i6, 1x, 8f12.5)
      ! -----------------------------------
    end

    ! subroutine calcAET(iy, im, id, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
    subroutine calcAET(model_dates, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
      use byhw_constants, only: START_YR_AET, END_YR_AET, YEAR, MONTH, DAY
      implicit none

      ! Arguments
      integer, intent(in) :: model_dates(:, :)
      integer, intent(in) :: nday
      real, intent(in) :: sim(:, :)
      integer, intent(in) :: NRUNS
      character(len=*), intent(in) :: HRUid(:)
      real, intent(in) :: pct_area(:)
      real, intent(inout) :: prmsOF

      ! Local variables
      logical, save :: first_call = .true.

      character :: header

      integer :: ihru
      integer :: imth
      integer :: ios
      integer :: iyr
      integer :: j
      integer :: m
      integer :: n
      integer, save :: nhru
        !! Number of HRUs in the headwater
      integer :: nm
      integer :: nmths
      integer :: nn
      integer :: ny

      real :: AET(3)
      real :: max_val
      real :: min_val

      ! Allocatable arrays
      integer, allocatable, save :: month_num(:)
      integer, allocatable, save :: nmthsim(:)
      real, allocatable, save :: mth(:, :, :)
      real, allocatable, save :: mthsim(:, :)

      ! ---------------------------------------------------------------------
      ! All 3 AET outputs available calendar years 2000-2010
      ! --------------------------
      ! --------------------------
      if (first_call) then
        first_call = .false.

        nhru = size(HRUid)

        ! Get AET outputs for HRU -- MWBM(in) MOD16(in) SSEBop(in)
        do ihru = 1, nhru
          close(20)
          open(20, file='./AET/HRU_'//HRUid(ihru), status='old')
          read(20, *) header

          ! First get the total number of records for the first HRU
          ! Allocate arrays based on this
          if (ihru == 1) then
            nmths = 0

            do
              read(20, *, iostat=ios) ny, nm
              if (ios /= 0) exit

              if (ny < START_YR_AET) cycle
              if (ny > END_YR_AET) exit
              nmths = nmths + 1
            end do

            rewind(20)
            read(20, *) header

            allocate(month_num(nmths))
            allocate(nmthsim(nmths))
            allocate(mth(nhru, 2, nmths))
            allocate(mthsim(nhru, nmths))
          end if

          nmths = 0

          do
            read (20, *, iostat=ios) ny, nm, (AET(j), j=1, size(AET, 1))
            if (ios /= 0) exit

            if (ny < START_YR_AET) cycle
            if (ny > END_YR_AET) exit

            min_val = 99999.0
            max_val = -99999.0

            do j = 1, size(AET, 1)
              if (AET(j) >= 0.0) then
                min_val = min(min_val, AET(j))
                max_val = max(max_val, AET(j))
                ! if (min > AET(j)) min = AET(j)
                ! if (max < AET(j)) max = AET(j)
              end if
            end do

            nmths = nmths + 1
            mth(ihru, 1, nmths) = min_val
            mth(ihru, 2, nmths) = max_val
            month_num(nmths) = nm
          end do

          close (20)
        end do
      end if    ! first_call

      ! --------------------------
      ! --------------------------
      ! Init array
      mthsim = 0.0

      do ihru = 1, nhru
        ! Init array
        nmthsim = 0
        nn = 0

        do n = 1, nday
          if (model_dates(YEAR, n) < START_YR_AET) cycle
          if (model_dates(YEAR, n) > END_YR_AET) exit

          if (model_dates(DAY, n) == 1) nn = nn + 1
          ! m = model_dates(MONTH, n)
          month_num(nn) = model_dates(MONTH, n)
          mthsim(ihru, nn) = mthsim(ihru, nn) + sim(ihru, n)
          nmthsim(nn) = nmthsim(nn) + 1
        end do

        do m = 1, nn
          mthsim(ihru, m) = mthsim(ihru, m) / real(nmthsim(m))
        end do
      end do

      ! ----------------------------
      ! ----------------------------
      !  NRMSE mth:
      prmsOF = 0.0

      do ihru = 1, nhru
        prmsof = prmsof + nrmse_range(mthsim(ihru, :), mth(ihru, 2, :), mth(ihru, 1, :), nn, pct_area(ihru))
      end do

      ! ----------------------------
      ! ----------------------------
      if (NRUNS == 0) then
        do ihru = 1, nhru
          close (75)
          open (75, file='./RESULTS/'//'AET_HRU'//HRUid(ihru), status='replace')
          write (75, '(a)') '  N YEAR MTH       AETlow     AEThigh        PRMS'
          iyr = 2000
          imth = 0

          do m = 1, nn
            imth = imth + 1

            if (imth == 13) then
              imth = 1
              iyr = iyr + 1
            end if

            write (75, 2) m, iyr, imth, mth(ihru, 1, m), mth(ihru, 2, m), mthsim(ihru, m)
          end do
        end do
      end if

      2 format(i3, 1x, i4, 1x, i3, 1x, 8f12.5)
    end

    ! =============================
    ! subroutine calcSCA(hw_str, iy, im, id, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
    subroutine calcSCA(hw_str, model_dates, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
      use byhw_constants, only: START_YR_SCA, END_YR_SCA, YEAR, MONTH, DAY
      implicit none

      ! Arguments
      character(len=*), intent(in) :: hw_str
      integer, intent(in) :: model_dates(:, :)
      integer, intent(in) :: nday
      real, intent(in) :: sim(:, :)
      integer, intent(in) :: NRUNS
      character(len=*), intent(in) :: HRUid(:)
      real, intent(in) :: pct_area(:)
      real, intent(inout) :: prmsOF

      ! Constants
      integer, parameter :: LOWER = 1
        !! Index for lower bound of obs SCA value
      integer, parameter :: UPPER = 2
        !! Index for upper bound of obs SCA value

      ! Local variables
      logical, save :: first_call = .true.

      integer :: cday
        !! Current day
      integer :: funit_obs
      integer :: i
      integer :: ihru
      integer :: ios
      integer :: n
      integer :: nd
      integer, save :: nhru
        !! Number of HRUs in the headwater
      integer :: nm
      integer :: ny
      real :: CI
        !! Clear index from MOD10C1 product
      real :: DIFF2
      real :: obs
      real :: RMSD
      real :: SUMDIFF

      ! Allocatable arrays
      real, allocatable, save :: obsSCA(:, :, :)

      ! ---------------------------------------------------------------------
      ! One SCA output available with CI for calendar years 2000-2010
      ! Values are between 0 and 1
      ! Calculate a min and max based on SCA and CI

      ! --------------------------
      ! --------------------------
      ! if (NRUNS == 1) then
      if (first_call) then
        first_call = .false.

        nhru = size(HRUid)

        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~
        ! Read the first SCA HRU to get the number of obs for array allocation
        open (newunit=funit_obs, file='./SCA/HRU_'//HRUid(1), status='old')
        read(funit_obs, *)   ! Skip the header

        n = 0

        do
          read(funit_obs, *, iostat=ios) ny
          if (ios /= 0) exit

          if (ny < START_YR_SCA) cycle
          if (ny > END_YR_SCA) exit

          n = n + 1
        end do

        close(funit_obs)

        ! Allocate the array
        allocate(obsSCA(nhru, 2, n))

        ! Init the array
        obsSCA = -888.0
        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~

        do ihru = 1, nhru
          ! Get SCA outputs for HRU -- 2 values: daily value and CI
          ! Only use value when CI > 70%
          open (newunit=funit_obs, file='./SCA/HRU_'//HRUid(ihru), status='old')
          read(funit_obs, *)   ! Skip the header

          n = 1

          do
            read(funit_obs, *, iostat=ios) ny, nm, nd, obs, CI
            if (ios /= 0) exit

            if (ny < START_YR_SCA) cycle
            if (ny > END_YR_SCA) exit

            if (CI >= 70.0) then
              obsSCA(ihru, LOWER, n) = (CI / 100.0) * obs
              obsSCA(ihru, UPPER, n) = obsSCA(ihru, LOWER, n) + (100.0 - CI) / 100.0

              if (nm == 8 .or. nm == 7) then
                ! Set to zero if July or August
                obsSCA(ihru, LOWER, n) = 0.0
                obsSCA(ihru, UPPER, n) = 0.0
              end if
            end if

            n = n + 1
          end do
          close (funit_obs)

          if (minval(obsSCA(ihru, LOWER, :)) == -888.0 .and. maxval(obsSCA(ihru, LOWER, :)) == -888.0) then
            write(*, '(a, a, a, a, a)') 'HW', hw_str, ' WARNING: HRU_', HRUid(ihru), &
                                        ' has no observed SCA meeting criteria; it will not be included in the OF calculations.'
          end if
        end do
      end if    ! first_call

      ! --------------------------
      ! --------------------------
      prmsof = 0.0
      do ihru = 1, nhru
        !  1--OBS
        !  2--SIM

        ! Weight the daily values by the lower OBSsca

        ! NRMSE day:
        sumdiff = 0.0
        n = 0
        i = 0

        ! WARNING: This is different from nrmse_range() in computing sumdiff
        do cday = 1, nday
          if (model_dates(YEAR, cday) < START_YR_SCA) cycle
          if (model_dates(YEAR, cday) > END_YR_SCA) exit
          i = i + 1

          if (obsSCA(ihru, 1, i) >= 0.0) then
            ! NOTE: PAN - I don't understand the math here; should look into it
            ! diff2 = ranged_diff(sim(ihru, cday), upper, lower) * (obsSCA(ihru, 1, i) + 1.0)
            diff2 = ranged_diff(sim(ihru, cday), obsSCA(ihru, UPPER, i), obsSCA(ihru, LOWER, i)) * (obsSCA(ihru, LOWER, i) + 1.0)
            sumdiff = sumdiff + (diff2 * (obsSCA(ihru, LOWER, i) + 1.0))
            n = n + 1
          end if
        end do

        if (n > 0) then
          rmsd = sqrt(sumdiff / real(n))
          prmsof = prmsof + (rmsd * pct_area(ihru))
        end if
      end do

      if (NRUNS == 0) then
        do ihru = 1, nhru
          close (75)
          open (75, file='./RESULTS/'//'SCA_HRU'//HRUid(ihru), status='replace')
          write (75, '(a)') '     n YEAR MTH DAY       SCAmin      SCAmax        PRMS'

          i = 0

          do cday = 1, nday
            if (model_dates(YEAR, cday) < START_YR_SCA) cycle
            if (model_dates(YEAR, cday) > END_YR_SCA) exit
            ! if (model_dates(YEAR, cday) >= START_YR_SCA .and. model_dates(YEAR, cday) <= END_YR_SCA) then
              i = i + 1
              write (75, 2) i, model_dates(YEAR, cday), model_dates(MONTH, cday), model_dates(DAY, cday), &
                            obsSCA(ihru, LOWER, i), obsSCA(ihru, UPPER, i), sim(ihru, cday)
            ! end if
          end do
        end do
      end if

      2 format(i6, 1x, i4, 1x, i3, 1x, i3, 1x, 8f12.5)
      return
    end

    ! ========================================
    ! subroutine calcRCH(iy, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
    subroutine calcRCH(model_dates, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
      use byhw_constants, only: START_YR_RCH, END_YR_RCH, YEAR, MONTH, DAY
      implicit none

      ! Arguments
      integer, intent(in) :: model_dates(:, :)
      integer, intent(in) :: nday
      real, intent(in) :: sim(:, :)
      integer, intent(in) :: NRUNS
      character(len=*), intent(in) :: HRUid(:)
      real, intent(in) :: pct_area(:)
      real, intent(inout) :: prmsOF

      ! Local variables
      logical, save :: first_call = .true.

      character :: header

      integer :: ihru
      integer :: ios
      integer :: j
      integer :: m
      integer :: n
      integer, save :: nhru
        !! Number of HRUs in the headwater
      integer :: nn
      integer, save :: numRCH
      integer :: ny

      real :: max_val
      real :: min_val

      ! Allocatable arrays
      real, allocatable, save :: annRCHobs(:, :, :)
      real, allocatable, save :: annRCHsim(:, :)

      ! Two RCH outputs available for calendar years 2000-2009

      ! --------------------------
      ! 'observed' RCH is already normalized using min and max for years
      ! 2000-2009 (10 years)
      ! --------------------------
      if (first_call) then
        first_call = .false.

        nhru = size(HRUid)

        do ihru = 1, nhru
          ! Get RCH outputs for HRU -- REITZ WATERGAP
          close (20)
          open (20, file='./RCH/HRU_'//HRUid(ihru), status='old')
          read (20, *) header

          if (ihru == 1) then
            numRCH = 0

            do
              read(20, *, iostat=ios) ny
              if (ios /= 0) exit

              if (ny < START_YR_RCH) cycle
              if (ny > END_YR_RCH) exit
              numRCH = numRCH + 1
            end do

            rewind(20)
            read(20, *) header

            ! Allocate arrays
            allocate(annRCHobs(nhru, 2, numRCH))
            allocate(annRCHsim(nhru, numRCH))
          end if

          do n = 1, numRCH
            read (20, *) ny, (annRCHobs(iHRU, j, n), j=1, 2)
          end do

          close (20)
        end do
      end if    ! first_call

      ! --------------------------
      ! --------------------------
      ! Init array
      annRCHsim = 0.0

      do ihru = 1, nhru
        do n = 1, nday
          if (model_dates(YEAR, n) < START_YR_RCH) cycle
          if (model_dates(YEAR, n) > END_YR_RCH) exit

          nn = model_dates(YEAR, n) - (START_YR_RCH - 1)
          annRCHsim(ihru, nn) = annRCHsim(ihru, nn) + sim(ihru, n)
        end do

        ! ----------------------------
        !  get min and max
        min_val = minval(annRCHsim(ihru, :))
        max_val = maxval(annRCHsim(ihru, :))

        if (max_val == min_val) max_val = max_val + (0.01 * max_val)
        if (max_val == min_val) max_val = max_val + 0.0001

        do j = 1, nn
          annRCHsim(ihru, j) = (annRCHsim(ihru, j) - min_val) / (max_val - min_val)
        end do
      end do

      ! ----------------------------
      !  NRMSE:
      prmsOF = 0.0

      do ihru = 1, nhru
        prmsof = prmsof + nrmse_range(annRCHsim(ihru, :), annRCHobs(ihru, 2, :), annRCHobs(ihru, 1, :), nn, pct_area(ihru))
      end do

      if (NRUNS == 0) then
        do ihru = 1, nhru
          close (75)
          open (75, file='./RESULTS/'//'RCH_HRU'//HRUid(ihru), status='replace')
          write (75, '(a)') 'YEAR       RCHmin      RCHmax        PRMS'

          do m = 1, nn
            write (75, 2) m + (START_YR_RCH - 1), annRCHobs(ihru, 1, m), annRCHobs(ihru, 2, m), annRCHsim(ihru, m)
          end do
        end do
      end if

      return
      2 format(i4, 1x, 8f12.5)
    end

    ! subroutine calcSOM(iy, im, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
    subroutine calcSOM(model_dates, nday, sim, NRUNS, HRUid, pct_area, prmsOF)
      use byhw_constants, only: START_YR_SOM, END_YR_SOM, YEAR, MONTH
      implicit none

      ! Arguments
      integer, intent(in) :: model_dates(:, :)
      integer, intent(in) :: nday
      real, intent(in) :: sim(:, :)
      integer, intent(in) :: NRUNS
      character(len=*), intent(in) :: HRUid(:)
      real, intent(in) :: pct_area(:)
      real, intent(inout) :: prmsOF

      ! Local variables
      logical, save :: first_call = .true.

      character :: header

      integer :: ihru
      integer :: ios
      integer :: iyr
      integer :: j
      integer :: m
      integer :: n
      integer, save :: nhru
        !! Number of HRUs in the headwater
      integer :: nm
      integer :: nn

      real :: jj
      real :: max_val
      real :: maxmin
      real :: min_val
      real :: RMSD
      real :: SUMDIFF

      ! Allocatable arrays
      real, allocatable, save :: mthSOMobs(:, :, :, :)
      real, allocatable, save :: mthSOMsim(:, :, :)

      ! --------------------------------------------------------------
      ! Min and max and 4 SOM outputs available calendar years 1982-2010
      ! NOTE there are more years but normalized using 1982-2010

      ! 'observed' SOM is already normalized annually and monthly
      ! with min and max listed first
      ! years 1982-2010
      ! --------------------------
      if (first_call) then
        first_call = .false.

        nhru = size(HRUid)

        do ihru = 1, nhru
          ! Get monthly SOM outputs for HRU
          ! 1982 is first year
          close(20)
          open(20, file='./SOM/mthHRU_'//HRUid(ihru), status='old')
          read(20, *) header

          if (ihru == 1) then
            iyr = 0
            do
              read(20, *, iostat=ios) n, nm
              if (ios /= 0) exit

              ! It is assumed that the exact date range is in the file(s)
              ! if (n < START_YR_SOM) cycle
              ! if (n > END_YR_SOM) exit

              if (nm == 1) iyr = iyr + 1
            end do

            rewind(20)
            read(20, *) header

            ! Allocate the arrays
            allocate(mthSOMobs(nhru, 2, 12, iyr))
            allocate(mthSOMsim(nhru, 12, iyr))
          end if

          do iyr=1, size(mthSOMobs, 4)
            do nm = 1, 12
              ! if (nm == 1) iyr = iyr + 1

              ! gfortran still tries to write to the array when it hits EOF
              if (iyr > size(mthSOMobs, 4)) exit

              read (20, *, iostat=ios) n, m, (mthSOMobs(ihru, j, nm, iyr), j=1, 2)
              if (ios /= 0) exit
            end do
          end do
          ! go to 50
          close (20)
        end do
      end if    ! first_call

      ! --------------------------
      ! --------------------------
      ! Init array
      mthSOMsim = 0.0

      do ihru = 1, nhru
        do n = 1, nday
          if (model_dates(YEAR, n) < START_YR_SOM) cycle
          if (model_dates(YEAR, n) > END_YR_SOM) exit

          nn = model_dates(YEAR, n) - (START_YR_SOM - 1)
          m = model_dates(MONTH, n)
          mthSOMsim(ihru, m, nn) = mthSOMsim(ihru, m, nn) + sim(ihru, n)
        end do

        ! ----------------------------
        !  get min and max for normalizing simulated over 1982-2010
        do m = 1, 12
          min_val = min(999999.0, minval(mthSOMsim(ihru, m, :)))
          max_val = max(-999999.0, maxval(mthSOMsim(ihru, m, :)))
          if (max_val == min_val) max_val = max_val + (0.01 * max_val)
          if (max_val == min_val) max_val = max_val + 0.0001

          do j = 1, nn
            mthSOMsim(ihru, m, j) = (mthSOMsim(ihru, m, j) - min_val) / (max_val - min_val)
          end do
        end do
      end do

      ! ----------------------------
      ! NRMSE month:
      prmsOF = 0.0
      do ihru = 1, nhru
        SUMDIFF = 0.0
        jj = 0.0

        do iyr = 1, nn
          do m = 1, 12
            sumdiff = sumdiff + ranged_diff(mthSOMsim(ihru, m, iyr), mthSOMobs(ihru, 2, m, iyr), mthSOMobs(ihru, 1, m, iyr))
            jj = jj + 1.0
          end do
        end do

        min_val = min(9999.0, minval(mthSOMobs(ihru, 1, :, :)))
        max_val = max(-9999.0, maxval(mthSOMobs(ihru, 2, :, :)))

        RMSD = sqrt(SUMDIFF / jj)

        maxmin = (max_val - min_val)

        if (maxmin < 1.0) maxmin = 1.0
        prmsOF = prmsOF + ((RMSD / (maxmin)) * pct_area(ihru))
      end do

      if (NRUNS == 0) then
        do ihru = 1, nhru
          close (75)
          open (75, file='./RESULTS/'//'SOM_HRU'//HRUid(ihru), status='replace')
          write (75, '(a)') 'YEAR MTH       SOMmin      SOMmax        PRMS'

          do iyr = 1, nn
            do m = 1, 12
              write (75, 2) iyr + (START_YR_SOM - 1), m, mthSOMobs(ihru, 1, m, iyr), mthSOMobs(ihru, 2, m, iyr), &
                            mthSOMsim(ihru, m, iyr)
            end do
          end do
        end do
      end if

      return
      2 format(i4, 1x, i3, 1x, 8f12.5)
    end

    pure function nmrse(sim_vals, obs_vals, num_times, weight) result(res)
      implicit none

      real :: res
      real, intent(in) :: sim_vals(:)
        !! Data array
      real, intent(in) :: obs_vals(:)
        !! Array of median values
      integer, intent(in) :: num_times
        !! Total number of observations over date range
      real, intent(in) :: weight

      ! Local variables
      integer :: m
      integer :: num_vals
      real :: max_val
      real :: maxmin
      real :: min_val
      real :: sumdiff
      real :: rmsd
      real :: diff2

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE of sim_vals with obs_vals:
      res = -999999.0
      sumdiff = 0.0

      num_vals = size(sim_vals, 1)  ! Get size of second dimension

      do m = 1, num_vals
        diff2 = (sim_vals(m) - obs_vals(m))**2
        sumdiff = sumdiff + diff2
      end do

      rmsd = sqrt(sumdiff / real(num_times))

      min_val = minval(obs_vals)
      max_val = maxval(obs_vals)

      maxmin = (max_val - min_val)
      if (maxmin < 1.0) maxmin = 1.0

      ! Return computed NRMSE
      res = (rmsd / maxmin) * weight
    end function

    pure function nrmse_range(darray, max_vals, min_vals, num_times, weight) result(res)
      implicit none

      real :: res
      real, intent(in) :: darray(:)   ! (4, 600)
        !! Data array
      real, intent(in) :: max_vals(:)
        !! Array of maximum values
      real, intent(in) :: min_vals(:)
        !! Array of minimum values
      integer, intent(in) :: num_times
        !! Total number of observations over date range
      real, intent(in) :: weight
      ! integer :: nn

      ! Local variables
      integer :: m
      integer :: num_vals
      real :: max_val
      real :: maxmin
      real :: min_val
      real :: sumdiff
      real :: rmsd
      real :: diff2

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE of darray with RANGE:
      res = -999999.0
      sumdiff = 0.0

      num_vals = size(darray, 1)  ! Get size of second dimension

      do m = 1, num_vals
        diff2 = ranged_diff(darray(m), max_vals(m), min_vals(m))
        sumdiff = sumdiff + diff2
      end do

      min_val = minval(min_vals)
      max_val = maxval(max_vals)

      rmsd = sqrt(sumdiff / real(num_times))
      maxmin = (max_val - min_val)
      if (maxmin < 1.0) maxmin = 1.0

      ! Return computed NRMSE
      res = (rmsd / maxmin) * weight
    end

    pure function nmrse_median(darray, median_vals, num_times, weight) result(res)
      implicit none

      real :: res
      real, intent(in) :: darray(:)
        !! Data array
      real, intent(in) :: median_vals(:)
        !! Array of median values
      integer, intent(in) :: num_times
        !! Total number of observations over date range
      real, intent(in) :: weight

      ! Local variables
      integer :: m
      integer :: num_vals
      real :: max
      real :: maxmin
      real :: min
      real :: sumdiff
      real :: rmsd
      real :: diff2

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE of darray with MEDIAN:
      res = -999999.0
      min = 9999.0
      max = -9999.0
      sumdiff = 0.0

      num_vals = size(darray, 1)  ! Get size of second dimension

      do m = 1, num_vals
        diff2 = (darray(m) - median_vals(m))**2
        if (min > median_vals(m)) min = median_vals(m)
        if (max < median_vals(m)) max = median_vals(m)
        sumdiff = sumdiff + diff2
      end do

      rmsd = sqrt(sumdiff / real(num_times))
      maxmin = (max - min)
      if (maxmin < 1.0) maxmin = 1.0

      ! Return computed NRMSE
      res = (rmsd / maxmin) * weight
    end function

    pure function nmrse_median_efc_high(darray, median_vals, efc, weight) result(res)
      implicit none

      real :: res
      real, intent(in) :: darray(:)
        !! Data array
      real, intent(in) :: median_vals(:)
        !! Array of median values
      integer, intent(in) :: efc(:)
        !! Array of EFC values
      real, intent(in) :: weight

      ! Local variables
      integer :: m
      integer :: num_vals
      real :: max
      real :: maxmin
      real :: min
      real :: sumdiff
      real :: rmsd
      real :: diff2

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE of darray with MEDIAN:
      res = -999999.0
      min = 9999.0
      max = -9999.0
      sumdiff = 0.0

      num_vals = size(darray, 1)  ! Get size of second dimension

      do m = 1, num_vals
        if (efc(m) <= 3) then
          diff2 = (darray(m) - median_vals(m))**2

          if (min > median_vals(m)) min = median_vals(m)
          if (max < median_vals(m)) max = median_vals(m)
          sumdiff = sumdiff + diff2
        end if
      end do

      rmsd = sqrt(sumdiff / real(count(efc <= 3)))
      maxmin = (max - min)
      if (maxmin < 1.0) maxmin = 1.0

      ! Return computed NRMSE
      res = (rmsd / maxmin) * weight
    end function

    pure function nmrse_median_efc_low(darray, median_vals, efc, weight) result(res)
      implicit none

      real :: res
      real, intent(in) :: darray(:)
        !! Data array
      real, intent(in) :: median_vals(:)
        !! Array of median values
      integer, intent(in) :: efc(:)
        !! Array of EFC values
      real, intent(in) :: weight

      ! Local variables
      integer :: m
      integer :: num_vals
      real :: max
      real :: maxmin
      real :: min
      real :: sumdiff
      real :: rmsd
      real :: diff2

      ! ~~~~~~~~~~~~~~~~~~~~~~
      ! NRMSE of darray with MEDIAN:
      res = -999999.0
      min = 9999.0
      max = -9999.0
      sumdiff = 0.0

      num_vals = size(darray, 1)  ! Get size of second dimension

      do m = 1, num_vals
        if (efc(m) > 3) then
          diff2 = (darray(m) - median_vals(m))**2

          if (min > median_vals(m)) min = median_vals(m)
          if (max < median_vals(m)) max = median_vals(m)
          sumdiff = sumdiff + diff2
        end if
      end do

      rmsd = sqrt(sumdiff / real(count(efc > 3)))
      maxmin = (max - min)
      if (maxmin < 1.0) maxmin = 1.0

      ! Return computed NRMSE
      res = (rmsd / maxmin) * weight
    end function

    pure function ranged_diff(val, upper, lower) result(res)
      implicit none
      ! Compute the difference of a value compared to a range

      real :: res
      real, intent(in) :: val
      real, intent(in) :: upper
      real, intent(in) :: lower

      res = 0.0
      if (upper == lower) then
        res = (val - upper)**2
      else
        if (val > upper) then
          res = (val - upper)**2
        end if

        if (val < lower) then
          res = (val - lower)**2
        end if
      end if
    end function

end module
