# Literature check: are gridded runoff products biased low vs. gauged streamflow in the PNW / mountain West?
(2026-09-25, subagent web search; [V]=source read, [S]=snippet only, [I]=inference)

Short answer: no blanket "biased low" verdict for ERA5-Land ro, GLDAS-2.1 Noah Qs+Qsb, or MWBM at monthly scale in the PNW.
The one PNW-specific large-sample LSM-vs-gauge study (Safeeq et al. 2014, J. Hydromet 15:2501, doi:10.1175/JHM-D-13-0198.1) finds the SIGN of bias flips with geology:
under-prediction in groundwater-dominated (High Cascades) basins (summer PBIAS -13%, Q5 -51%), over-prediction in runoff-dominated basins (summer PBIAS +48%, Q5 +19%).

Key sources
- ERA5-Land: Munoz-Sabater et al. 2021 ESSD doi:10.5194/essd-13-4349-2021 [V] (negative snow bias at mountain sites from orography smoothing); Bain et al. 2023 J Hydrol doi:10.1016/j.jhydrol.2022.128624 [S]; Boulange et al. 2026 EGUsphere doi:10.5194/egusphere-2026-2739 [V].
- GLDAS-2.1 Noah: Duvvuri et al. 2024 Sci Rep doi:10.1038/s41598-024-75361-w [V] (poor in snow-dominated/semi-arid/regulated; timing early, magnitude over); Lv et al. 2018 Water doi:10.3390/w10080969 [V] (June-July melt peak too high); Zaitchik et al. 2010 WRR doi:10.1029/2009WR007811 [V]; Broxton et al. 2016 J Hydromet doi:10.1175/JHM-D-16-0056.1 [S] (all LDAS underestimate western SWE); Xia et al. 2012 JGR doi:10.1029/2011JD016051 [V].
- MWBM: Bock et al. 2016 HESS doi:10.5194/hess-20-2861-2016 [V] (median monthly NSE 0.76 at 1575 gauges); Hostetler & Alder 2016 WRR doi:10.1002/2016WR018665 [V]; McCabe & Wolock 2011 WRR doi:10.1029/2011WR010630 [S]; Henn et al. 2018 J Hydrol doi:10.1016/j.jhydrol.2017.03.008 [S] (5-60% spread in gridded precip in western complex terrain). No PNW evaluation of the nClimGrid-driven MWBM found.
- PNW / groundwater: Safeeq et al. 2014 [V]; Wenger et al. 2010 WRR doi:10.1029/2009WR008839 [V]; Tague et al. 2008 Clim Change doi:10.1007/s10584-007-9294-8; Tague & Grant 2009 WRR doi:10.1029/2008WR007179 [V]; Safeeq et al. 2014b HESS 18:3693 [S]; Su et al. 2024 HESS doi:10.5194/hess-28-3079-2024 [V]; Towler et al. 2023 HESS doi:10.5194/hess-27-1809-2023 [V] (NWM/NHM OVER-estimate western volumes at non-reference gages); Beck et al. 2017 HESS doi:10.5194/hess-21-2881-2017 [V]; Putman et al. 2024 HESS doi:10.5194/hess-28-2895-2024 [V].
- Bias vs attributes: Husic et al. 2025 HESS doi:10.5194/hess-29-4457-2025 [V] (RF+Shapley, 48 attributes); van der Heijden et al. 2026 WRR doi:10.1029/2025WR040375 [V] (NWM underestimates baseflow; snow fraction & timing dominate in snow basins).

Synthesis
- Expected monthly SHAPE for products with no deep aquifer (all three of ours): low Jul-Oct, high Dec-Mar in High Cascades; annual totals closer to unbiased unless crest precip undercaught (most plausible for 0.25 deg GLDAS and station-based nClimGrid).
- Covariates with literature support, in order: geology/permeability or baseflow index; snow fraction and melt timing; annual precip / runoff ratio / aridity; elevation; regulation/irrigation status. Forest cover appears in no evaluation as an explanatory covariate.
- Gap: no paper evaluates ERA5-Land ro, GLDAS-2.1 Qs+Qsb, or nClimGrid MWBM directly against Oregon gauges.
