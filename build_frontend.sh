#!/bin/bash
# Script to build the React frontend and copy to the correct location

set -e

echo "Building React frontend..."
cd frontend
npm install
npm run build

echo "Copying build files to static directory..."
rm -rf ../static/react
mkdir -p ../static/react
cp -r build/* ../static/react/

echo "Frontend build complete!"
