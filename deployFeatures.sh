#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the repository (assumes the script runs from within the repo)
REPO_DIR=$(git rev-parse --show-toplevel)
cd "$REPO_DIR"

# Ensure branch is correct
BRANCH_NAME=$(git symbolic-ref --short HEAD)
if [ "$BRANCH_NAME" != "master" ]; then
  echo "Switching to the master branch..."
  git checkout master
fi

# Create directories if they don't exist
mkdir -p css js

# Create and edit index.html
echo '<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" type="text/css" href="css/home.css">
</head>
<body>
  <script src="js/home.js"></script>
</body>
</html>' > index.html

# Create and edit home.css
echo 'body {
  background-color: #f0f0f0;
  font-family: Arial, sans-serif;
}' > css/home.css

# Create and edit home.js
echo 'let stressLevel = 30;
console.log("Stress Level:", stressLevel);' > js/home.js

# Add the files to the repository
git add index.html css/home.css js/home.js

# Commit the changes
git commit -m "Update project files with deploy script"

# Push to GitHub
git push origin master
