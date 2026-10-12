#!/bin/sh
# Download Fed-ISIC2019's raw files and verify them BEFORE FLamby's script
# unzips the images, deletes the zip and rewrites the CSVs (after that they
# can no longer be checked). Accept the ISIC 2019 (CC BY-NC 4.0) and HAM10000
# terms first; see README.
#
# Checks (expected values from the server's headers, 2026-10-12):
#   - CSVs: MD5 equals the server's ETag (single-part uploads, so ETag = MD5)
#   - zip: exact byte size, and `unzip -t` (CRC-32 of every file in the zip).
#     The zip's ETag ("...-569") is a multipart hash, not an MD5 of the file,
#     so it can't be compared without knowing the upload's part size.
#   - zip holds 25,331 images (ISIC 2019 training set)
# Stops at the first failure. Then run FLamby's script (it asks for the
# licence once and uses the verified files; see the command printed at the end).
set -e
cd "$(dirname "$0")"
OUT="$PWD/data/flamby/fed_isic2019"
BASE=https://isic-challenge-data.s3.amazonaws.com/2019
ZIP_BYTES=9771618190
META_MD5=8ca5fdef200ffd8579f48c823b504f7e
GT_MD5=2c02bdcc6e7f36d355f4f86b210595ae
N_IMAGES=25331
mkdir -p "$OUT"

fail() { echo "FAILED: $*" >&2; exit 1; }

for f in ISIC_2019_Training_Metadata.csv ISIC_2019_Training_GroundTruth.csv; do
  curl -fsSL --retry 3 "$BASE/$f" -o "$OUT/$f"
done
[ "$(md5 -q "$OUT/ISIC_2019_Training_Metadata.csv")" = "$META_MD5" ] || fail "metadata CSV MD5"
[ "$(md5 -q "$OUT/ISIC_2019_Training_GroundTruth.csv")" = "$GT_MD5" ] || fail "ground-truth CSV MD5"
echo "CSVs OK (MD5 match)"

ZIP="$OUT/ISIC_2019_Training_Input.zip"
if [ "$(stat -f%z "$ZIP" 2>/dev/null)" != "$ZIP_BYTES" ]; then
  echo "$(date '+%H:%M') downloading the images zip (9.8 GB); -C - resumes a partial file"
  curl -fL --retry 5 -C - "$BASE/ISIC_2019_Training_Input.zip" -o "$ZIP"
fi
[ "$(stat -f%z "$ZIP")" = "$ZIP_BYTES" ] || fail "zip size $(stat -f%z "$ZIP") != $ZIP_BYTES"
echo "$(date '+%H:%M') zip size OK; testing every file's CRC-32"
unzip -tq "$ZIP" || fail "unzip -t (CRC check)"
n=$(unzip -Z1 "$ZIP" | grep -c '\.jpg$')
[ "$n" = "$N_IMAGES" ] || fail "zip has $n images, expected $N_IMAGES"
echo "$(date '+%H:%M') zip OK: $n images, all CRCs match"

echo
echo "Next (answer FLamby's licence prompt with y):"
echo ".venv/bin/python third_party/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py --output-folder \"$OUT\""
