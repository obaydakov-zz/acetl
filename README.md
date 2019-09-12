# acetl
ACETL

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch read_s3.py" \
  --prune-empty --tag-name-filter cat -- --all