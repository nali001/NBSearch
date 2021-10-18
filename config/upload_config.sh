# Bash script for uploading local data to Surfdrive 
# Source: config/
# Destination: /home/NBSearch/config
SURF_CONFIG=WamWQK688VEXL02
for f in *; do
    if [ -f "$f" ]; then
    	echo "Upload $f"
    	curl -u $SURF_CONFIG:nana -T $f "https://surfdrive.surf.nl/files/public.php/webdav/"
    fi
done
