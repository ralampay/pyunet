nice python -m pip list --outdated --format=json | \
      jq -r '.[] | "\(.name)==\(.latest_version)"' | \
      xargs --no-run-if-empty -n1 pip3 install -U

#outputs the list of installed packages
pip freeze > requirements.txt
