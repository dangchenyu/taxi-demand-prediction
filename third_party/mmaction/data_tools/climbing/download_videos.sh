mkdir ../../data/climbing/original_videos
for line in `cat ../../data/climbing/videos_list.txt`
do
        youtube-dl https://www.youtube.com/watch?v=$line -f mp4 -o '../../data/climbing/original_videos/%(id)s.%(ext)s'
done