#!/bin/bash

# Script to copy clean files to new repo

echo "Copying clean files to ounass-api-pods..."

# Source and destination
SRC="/Users/root1/Desktop/Projects/ounass-api"
DEST="/Users/root1/Desktop/ounass-api-pods"

# Copy source code
echo "Copying source code..."
cp "$SRC/src/main.py" "$DEST/src/"
cp "$SRC/src/__init__.py" "$DEST/src/" 2>/dev/null || echo "" > "$DEST/src/__init__.py"
cp "$SRC/src/api/endpoints.py" "$DEST/src/api/"
cp "$SRC/src/api/__init__.py" "$DEST/src/api/" 2>/dev/null || echo "" > "$DEST/src/api/__init__.py"
cp "$SRC/src/models/forecasting.py" "$DEST/src/models/"
cp "$SRC/src/models/__init__.py" "$DEST/src/models/" 2>/dev/null || echo "" > "$DEST/src/models/__init__.py"
cp "$SRC/src/services/sheets_service.py" "$DEST/src/services/"
cp "$SRC/src/services/__init__.py" "$DEST/src/services/" 2>/dev/null || echo "" > "$DEST/src/services/__init__.py"

# Copy config files
echo "Copying config files..."
cp "$SRC/requirements.txt" "$DEST/"
cp "$SRC/.env.example" "$DEST/"
cp "$SRC/.gitignore" "$DEST/"
cp "$SRC/Dockerfile" "$DEST/"
cp "$SRC/run.sh" "$DEST/"

# Copy data directory if it exists
if [ -d "$SRC/data" ]; then
    echo "Copying sample data..."
    mkdir -p "$DEST/data"
    cp "$SRC/data/sample_data.csv" "$DEST/data/" 2>/dev/null || echo "No sample data found"
fi

# Create necessary directories
mkdir -p "$DEST/logs"
mkdir -p "$DEST/tests"

echo "✅ Done! Clean repository ready at $DEST"
echo ""
echo "Next steps:"
echo "1. cd $DEST"
echo "2. git init"
echo "3. git add ."
echo "4. git commit -m 'Initial commit: Clean OUNASS Pod Forecasting API'"
echo "5. git remote add origin https://github.com/sorted78/ounass-api-pods.git"
echo "6. git branch -M main"
echo "7. git push -u origin main"
