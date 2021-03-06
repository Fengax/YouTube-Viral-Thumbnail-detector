import urllib.request
from googleapiclient.discovery import build

viral_list = ["Viral", "Normal", "Shit"]

def calcViral(avgView, view, likeRatio):
    if view > avgView + avgView * 0.5 and likeRatio > 0.9:
        return 0
    elif view > avgView and likeRatio > 0.7:
        return 1
    else:
        return 2

api_key = "bruh"

youtube = build("youtube", "v3", developerKey=api_key)

request_channel = youtube.channels().list(
    part = "statistics",
    id = "UCtMVHI3AJD4Qk4hcbZnI9ZQ"
)

items = request_channel.execute()['items']
stats = items[0]["statistics"]
total_views = stats["viewCount"]
total_videos = stats["videoCount"]
sub_count = stats["subscriberCount"]

avg_view = int(total_views) // int(total_videos)

pageToken = ""
counter = 1000000
while True:
    print("New page")

    request_videos = youtube.search().list(
        part="snippet",
        channelId="UCtMVHI3AJD4Qk4hcbZnI9ZQ",
        maxResults="50",
        type="video",
        pageToken = pageToken
    )

    response = request_videos.execute()
    pageToken = response["nextPageToken"]

    for i in response["items"]:
        dict = i["snippet"]
        dict2 = i["id"]
        vid_id = dict2["videoId"]
        thumbnail = dict["thumbnails"]
        thumbnail_high = thumbnail["high"]
        url = thumbnail_high["url"]

        print(url)

        request_vidrating = youtube.videos().list(
            part = "statistics",
            id=vid_id
        )

        dict3 = request_vidrating.execute()["items"]
        stats = dict3[0]["statistics"]
        likeCount = stats["likeCount"]
        dislikeCount = stats["dislikeCount"]
        likeRatio = int(likeCount) / (int(likeCount) + int(dislikeCount))
        viewCount = stats["viewCount"]

        virality = calcViral(int(avg_view), int(viewCount), likeRatio)
        if virality == 0:
            counter += 1
            urllib.request.urlretrieve(url, "D:/images/viral/" + str(counter) + ".jpg")
            print("Stored a {} thumbnail".format(viral_list[virality]))
        elif virality == 1:
            counter += 1
            urllib.request.urlretrieve(url, "D:/images/normal/" + str(counter) + ".jpg")
            print("Stored a {} thumbnail".format(viral_list[virality]))
        else:
            counter += 1
            urllib.request.urlretrieve(url, "D:/images/bad/" + str(counter) + ".jpg")
            print("Stored a {} thumbnail".format(viral_list[virality]))
