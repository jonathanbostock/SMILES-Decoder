# Install git-filter-repo
pip install git-filter-repo

# Find large files
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -nr -k3 | head -n 10

# Remove specific large file
git filter-repo --path-glob '*.csv' --invert-paths --force