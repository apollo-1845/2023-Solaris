import time
# we will be using time instead of the tutorial's suggested EXIF data because the time function returns the time elapsed
#with significantly greater precision than the time recorded in the EXIF data.
start=time.process_time()
def checkTimeDiff(start):
    return time.process_time()-start