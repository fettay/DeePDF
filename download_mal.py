import os
from vt import VTConnection
from storage import FsStorage
import logging
from globals import DATAMALDIR
import schedule
import time
import datetime


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
vt_key = os.environ['VT_API_KEY']

conn = VTConnection(vt_key)

query = 'type:pdf positives:10+ fs:{}+ fs:{}-'
fname = 'benign_vt2.csv'

start = datetime.datetime(2019, 2, 15)
end = datetime.datetime(2019, 3, 15)
delta = end - start


def search_files():
#     for i in range(delta.days + 1):
#         date = (start + datetime.timedelta(days=i)).strftime('%Y-%m-%d')
#         date2 = (start + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d')
    conn.search(query.format('2018-06-30', '2018-09-29'), outfile=fname, max_result=100000)

def download_files():
    storage = FsStorage()
    with open(fname) as f:
        all_mals = f.read().split()
    for mal in all_mals:
        try:
            downloaded = storage.get('pdfs/' + mal)
            if len(downloaded) > 0:
                logger.info("Already downloaded %s" % mal)
                continue
        except Exception: # FileNotFound in bucket
            pass
        content = conn.download_file(mal, store=False)
        if len(content) == 0:
            logger.error('Consumed all VT')
            return
        storage.put('pdfs/' + mal, content)
        logger.info('Downloaded %s' % mal)



def main():
    schedule.every().day.at("10:00").do(download_files)
    while(True):
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
#     download_files()
    main()
#     search_files()


    