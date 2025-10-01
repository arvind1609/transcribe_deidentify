# Navigate to the directory containing your audio files
# CHANGE TO DIRECTORY WITH AUDIO DATA
cd ../data/

# Loop through all MP3 files in the current directory
for file in *.mp3; do
    # Get the filename without the extension (e.g., 'cbt_2' from 'cbt_2.mp3')
    base=$(basename "$file" .mp3)
    
    # Execute the ffmpeg command
    ffmpeg -i "$file" -ac 1 -ar 16000 "${base}.wav"
done