DIR=$1/*.tiff

for FILE in $DIR
do
  NEW_FILE="${FILE/_person\.ome\.tiff/\.tiff}"
  echo "Renaming $FILE to $NEW_FILE..."
  mv $FILE $NEW_FILE
done

