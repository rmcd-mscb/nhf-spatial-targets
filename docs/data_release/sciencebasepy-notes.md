# `sciencebasepy` capability notes

What the [`sciencebasepy`](https://github.com/DOI-USGS/sciencebasepy) Python
client makes easy, what it can't do, and the sharp edges we'll hit when
implementing `release/sb_client.py`. Captured 2026-05-26.

## TL;DR

**Easy:** Token/password auth, item CRUD with `parentId`, file upload
(single, multi, and via S3), file download, search by title/query, ACL
management.

**You build yourself:**

- **DOI minting** — not in the library; manual SB-staff step after IPDS
  approval.
- **FGDC XML generation** — not in the library; bring your own XML and
  attach as a file. The library can *scrape* FGDC from S3 via GraphQL but
  cannot emit it.
- **Find-by-title idempotency** — no dedicated `upsert_by_title` method; you
  use `find_items(q='...')` and adopt the matched ID into your registry.
- **Resumable upload** — single-file uploads are not resumable. Network
  drops mid-multi-GB-file mean full retry. Multipart S3 upload exists
  internally for cloud paths but is not exposed.
- **Version history** — re-uploading a file with the same name silently
  overwrites; no version history is retained. Track checksums in
  `catalog/release_registry.yml` to detect drift across builds.

## Authentication

| Method | Method signature | Use when |
|---|---|---|
| Token | `sb.add_token(token_json)` or `sb.get_token()` (browser flow) | Default — short-lived, scoped to user |
| Username + password | `sb.login(username, password)` or `sb.loginc(username)` (interactive) | Service accounts only; password auth is gated on USGS service-account roles |
| Status | `sb.is_logged_in()`, `sb.logout()` | Sanity checks |

For this pipeline: store the token under `.credentials.yml.sciencebase.token`,
materialize to whatever pickup location `SbSession` expects (likely
`~/.sb_token.json` — confirm against current `sciencebasepy` docs at PR-G
time) via an extended `materialize-credentials` command.

`get_token()` (browser flow) has had issues on Windows 11 launching the
default browser; document the manual token-paste fallback in the runbook.

## Item CRUD

| Method | Notes |
|---|---|
| `create_item(item_json)` | Requires `title` and `parentId`. Returns full item JSON including generated `id`. |
| `get_item(itemid, params={'fields': '...'})` | Field filtering supported via the `fields` param. |
| `update_item(item_json)` | Pass dict with `id` + modified fields. Returns updated item. |
| `delete_item(item_json)` | Returns boolean. |
| `move_item(item, new_parent_id)` | Re-parent an item. |
| `create_items()` / `update_items()` / `delete_items()` | Batch variants for efficiency. |

Item body is a free-form dict; SB doesn't validate FGDC at this layer.
Citation, time period, bounding box are populated by setting the right
top-level keys in the item JSON — there are no typed setters for them.

## File upload

| Method | Notes |
|---|---|
| `upload_file_to_item(item, filename, scrape_file=True)` | Single file. `scrape_file=True` attempts metadata extraction (variable results). |
| `upload_files_and_upsert_item(item_json, files=[...])` | Upserts the item and uploads files in one call. Matches existing item by `id`. |
| `upload_cloud_file_to_item(item, s3_url)` | S3 path; multipart upload happens internally. |
| `replace_file()` | In-place replace; updates checksum + dateUploaded + uploadedBy fields. |
| `publish_array_to_public_bucket(items, urls)` | Batch S3 publish. |

Sharp edges:

- **No exposed multipart**. The cloud upload path uses multipart internally,
  but the standard `upload_file_to_item` does not. Multi-GB files have
  occasionally been reported as failing on flaky connections (no GitHub
  issue numbers documented at capture time; verify before relying).
- **No size limit documented**. SB itself has practical limits ~10 GB per
  file, ~100 GB per item.
- **`scrape_file=True` for `.xml`** can create duplicate extension entries
  (sciencebasepy issue #17). Recommended: upload FGDC XML with
  `scrape_file=False` and rely on our own metadata population.

## Idempotency and lookup

The library has one upsert helper:

```python
sb.upload_files_and_upsert_item(item_json, files=[...])
```

This matches the existing item by `id` (must already be in `item_json`).
There is no `upsert_by_title()`. To implement that ourselves:

```python
results = sb.find_items(params={"q": canonical_title, "parentId": umbrella_id})
matches = [i for i in results["items"] if i["title"] == canonical_title]
if len(matches) == 1:
    item_id = matches[0]["id"]
    mode = "update"
elif len(matches) == 0:
    mode = "create"
else:
    raise SbMultipleMatches(...)
```

`find_items()` returns paginated results; iterate `["nextlink"]` for large
result sets. This is what `release/publish.py` will do on the
registry-says-missing path (see the idempotency strategy in the design plan).

## DOI

Not supported by the library at all. DOI minting is a ScienceBase-staff
operation. After mint, the DOI appears in `item["identifiers"][0]["key"]`
(format: `doi:10.5066/<id>`).

The pipeline workflow:

1. Operator runs `nhf-targets release publish --scope umbrella --confirm` —
   creates the parent item without DOI.
2. IPDS approval happens externally.
3. Operator emails `sciencebase@usgs.gov` requesting DOI mint.
4. After mint, operator copies the DOI into
   `catalog/release_registry.yml.umbrella.doi` and reruns the umbrella
   publish — this re-emits FGDC + ISO XMLs with the DOI populated and patches
   the item body.

## Versioning

Re-uploading a file with the same `name` to an existing item overwrites the
previous file. SB does not keep a version history; the only metadata that
changes is `dateUploaded`, `uploadedBy`, and `checksum`.

To detect drift between builds, our `release/registry.py` records
`checksums_sha256_of_manifest` (SHA256 of the staged `checksums.csv`) per
child. A build whose manifest checksum doesn't match the previous build is
flagged in `release status` and triggers full re-upload.

## Bulk operations and rate limiting

- `create_items()`, `update_items()`, `delete_items()` for batch CRUD.
- `download_cloud_files(urls, tokens)` for multi-file download via tokenized
  S3 links.
- `generate_S3_download_links()` to create download tokens.
- `publish_array_to_public_bucket()` for bulk S3 publish.

Rate limiting is not explicitly documented in the README or recent issues.
Our `sb_client.upload_file` wrapper retries 429 + 5xx with exponential
backoff (3 tries; 2s / 4s / 8s) to be defensive.

## Known sharp edges (from sciencebasepy GitHub issues at capture time)

| Issue | Behavior |
|---|---|
| #52 | `get_token()` may fail to open the browser on Windows 11; manual paste fallback needed. |
| #51 | SSL errors on S3 cloud uploads; sometimes mitigated by retry. |
| #17 | Uploading metadata XML with `scrape_file=True` creates duplicate extension entries. |
| #9 | Web-link items (no file payload, just a URL) aren't well-supported. |

If new issues surface during PR-E (SB client) implementation, link them
here. This file is the running log of "things to watch out for."

## Library facts worth memorizing

- **DOI minting**: not in the library. Manual SB-staff step.
- **FGDC XML generation**: not in the library. Bring your own XML.
- **Find by title**: roll your own with `find_items(q=...)`.
- **Resumable upload**: no.
- **Version history**: no — overwrite is silent. Track checksums.

These are also captured as a project memory under
`project_sciencebase_release_facts` so they don't get re-discovered.
