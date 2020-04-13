import requests
import itertools
from multiprocessing import Pool
from retry import retry


"""
Bunch of common request to vt and parsers
Set env variable vt_proxies = False to remove proxies
"""

#PROXIES = {'http': '194.29.34.106:8080', CP PROXIES
#           'https': '194.29.34.106:8080'}
PROXIES = {}
SEARCH_BATCH_SIZE = 300
VT_BATCH_REQUEST_SIZE = 10

DEFAULT_MASK = []
# DEFAULT_IGNORE = ['network']
DEFAULT_IGNORE = []


class VTConnection(object):
    """
    Store basic info for connection to VT
    """
    def __init__(self, api_key, proxies=PROXIES, timeout=10, verify=True, ssl=True):
        self.proxies = proxies
        self.api_key = api_key
        self.timeout = timeout
        self.verify = verify
        if ssl:
            self.http = 'https://'
        else:
            self.http = 'http://'

    @retry(tries=60, delay=10)
    def call(self, url, data=None, method='GET', params=None, all_info=0):
        """
        Perform a regular call to VT API
        :param url: URL of the request
        :param data:
        :param method: HTTP Method
`       :param params: Parameters of the request
        :param all_info: Boolean fetch all info
        :return json result
        """
        if params is None:
            params = {'apikey': self.api_key, 'resource': data, 'allinfo': all_info}
        else:
            params['apikey'] = self.api_key
        if self.proxies is not None:
            response = requests.get(url, params=params,
                                    proxies=self.proxies, timeout=self.timeout, verify=self.verify)
            if method.lower() == 'get':
                pass
            else:
                response = requests.get(url, params=params,
                                        proxies=self.proxies, timeout=self.timeout, verify=self.verify)
        else:
            if method.lower() == 'get':
                response = requests.get(url, params=params)
            else:
                response = requests.post(url, params=params)
        return response

    def _batch_report(self, batch, all_info, report_type, verbose):
        params = {'resource': batch, 'allinfo': all_info, 'hash': batch}
        try:
            response = self.call(self.http + 'www.virustotal.com/vtapi/v2/file/%s' % report_type, params=params)
        except Exception:
            print("Timeout error with batch %s" % batch)
            return
        if verbose:
            print("Batch Done")
        try:
            return response.json()
        except Exception:
            return []

    def get_report(self, hashes, report_type='report', all_info=1, verbose=False, nb_process=1):
        """
        Get report of a list of hashes in virus total and combines results
        :param hashes: List of hashes
        :param report_type: url to reach (report, behaviour...)
        :param all_info: Get all info
        :param verbose: Print at every batch done
        :param nb_process: Number of processes that launches requests
        :return list of results
        """
        if not isinstance(hashes, list):
            hashes = [hashes]
        if report_type == 'report':  # Batch request
            data = [",".join(hashes[i:i + VT_BATCH_REQUEST_SIZE]) for i in range(0, len(hashes), VT_BATCH_REQUEST_SIZE)]
        else:
            data = hashes

        if nb_process == 1:
            full_response = [self._batch_report(batch, all_info, report_type, verbose) for batch in data]
            full_response = [res for res in full_response if res is not None]

        else:
            with Pool(nb_process) as p:
                list_args = [(batch, all_info, report_type, verbose, self.proxies, self.http, self.api_key) for batch in data]
                full_response = p.map(_batch_report_async, list_args)

        if len(full_response) == 0:
            raise ValueError("No responses available for" + ",".join(hashes))
        if isinstance(full_response[0], list):
            full_response = list(itertools.chain(*full_response))
        return full_response

    def download_file(self, hash_, store=True, dir_dest="", file_ext=""):
        """
        Download a file from VT and stores it
        :param hash_: any hash of the file
        :param store: store as file
        :param dir_dest: destination directory
        :param file_ext: extensions of the files
        """
        url = self.http + 'www.virustotal.com/vtapi/v2/file/download'
        params = {'hash': hash_}
        response = self.call(url, params=params)
        if not store:
            return response.content
        f = open(dir_dest + hash_ + file_ext, 'wb')
        f.write(response.content)
        f.close()

    def search(self, query, store=True, outfile='result_search.txt', init_offset=None, max_result=None):
        """
        Implements a search request and retrieve the list of all hashes
        @param query: Content of the query
        @param store: (bool) store result in a file
        @param outfile: Name of the output file (if any)
        @param init_offset: Offset to start from (useful for recovery)
        """

        url = self.http + 'www.virustotal.com/vtapi/v2/file/search'
        params = {'query': query}
        if store:
            result = open(outfile, 'a+')
        else:
            result = []
        i = 1
        offset = init_offset
        if offset is not None:
            params['offset'] = offset
        while True:
            if max_result is not None and (i - 1)*300 > max_result:
                break
            try:
                response = self.call(url, params=params).json()
            except Exception as e:
                print(e)
                if offset is not None:
                    print("Broke at offset %s" % offset)
                break
            if response['response_code'] != 1:
                break
            print("Got %d files" % (SEARCH_BATCH_SIZE*i))
            if isinstance(result, list):
                result.extend(response['hashes'])
            else:
                result.write('\n'.join(response['hashes']) + '\n')
            if 'offset' not in response:
                break
            offset = response['offset']
            params['offset'] = offset
            i += 1

        if not store:
            return result


# Async function
def _batch_report_async(args):
    batch, all_info, report_type, verbose, proxies, http, api_key = args
    params = {'resource': batch, 'allinfo': all_info, 'hash': batch}
    vt_instance = VTConnection(api_key, proxies=proxies)
    try:
        response = vt_instance.call(http + 'www.virustotal.com/vtapi/v2/file/%s' % report_type, params=params)
    except Exception:
        print("Timeout error with batch %s" % batch)
        return
    if verbose:
        print("Batch Done")
    try:
        return response.json()
    except Exception:
        return []

# Function that extract data from results

def get_vendors_verdict(responses, mask=DEFAULT_MASK):
    """
    Parse the result and return a dict of tuples for the verdicts by hash
    Allows you to drop some verdict using mask
    :param responses: list of hashes data
    :param mask: list of Vendors to ignore
    :return dict on sha with a value a list of tuples: vendors, verdict
    """
    formatted = {}
    if not isinstance(responses, list):
        responses = [responses]
    for res in responses:
        if 'scans' in res:
            try:
                formatted[res['sha256']] = [(vendor, scan['result']) for vendor, scan in res['scans'].items()
                                            if scan['result'] is not None and vendor not in mask]
            except ValueError:
                print("Invalid Response")
        else:
            print("Missing scans for %s" % res['resource'])
    return formatted


def parse_syscalls(responses, ordered_hashes):
    """
    Parse the result and return a list of tuples for the behavior
    :param responses from vt api
    :param ordered_hashes list of all queried hashes
    :return  dict on sha with a value a list of tuples category, status, api_call
    """
    result = {}
    for hash_, response in zip(ordered_hashes, responses):
        if 'behavior' not in response or 'processes' not in response['behavior']:
            continue
        if len(response['behavior']['processes']) == 0 or 'calls' not in response['behavior']['processes'][0]:
            print(response['behavior']['processes'])
            continue
        res = []
        for bh in response['behavior']['processes'][0]['calls']:
            res.append((bh['category'], bh['status'], bh['api']))
        result[hash_] = res
    return result


def extract_from_dict(dict_data, res_list=None, to_ignore=DEFAULT_IGNORE):
    """
    Get all values from a nested dictionary
    :param dict_data: dictionnary of data (or string or list)
    :param res_list: list of results, set None for nothing
    :param to_ignore: list of keys to ignore
    """
    if res_list is None:
        res_list = []
    if to_ignore is None:
        to_ignore = []
    if isinstance(dict_data, str):
        res_list.append(dict_data)
        return
    if isinstance(dict_data, list):
        for v1 in dict_data:
            extract_from_dict(v1, res_list)
    elif isinstance(dict_data, dict):
        for k, v in dict_data.items():
            if k in to_ignore:
                continue
            if isinstance(v, dict):
                extract_from_dict(v, res_list)
            elif isinstance(v, list):
                for v1 in v:
                    extract_from_dict(v1, res_list)
            elif isinstance(v, str):
                res_list.append(v)
        return res_list


def select_nested_values(dict_data, keys, inplace=False):
    """
    Get a dict in input and value with nested keys [key1, key2.key_nested] and select only those values
    :param dict_data: dict
    :param keys: Formatted keys string
    :param inplace: BOOL Inplace dict processing
    :return: dict
    """
    if not isinstance(dict_data, dict):
        raise ValueError("Not a dictionnary")
    if not inplace:
        dict_data_copy = dict_data.copy()
    else:
        dict_data_copy = dict_data
    splitted = [k.split('.') for k in keys]
    keys_prim = [k[0] for k in splitted]
    to_drop = []
    for k, v in dict_data_copy.items():
        if isinstance(v, dict):
            nested = [spl[1] for spl in splitted if len(spl) > 1 and spl[0] == k]
            if len(nested) > 0:
                dict_data_copy[k] = select_nested_values(v, nested, inplace=False)
        if k not in keys_prim:
            to_drop.append(k)
    for k in to_drop:
        dict_data_copy.pop(k, '')
    return dict_data_copy


if __name__ == '__main__':
    import json
    conn = VTConnection(timeout=45)
    # conn.search('positives:1- fs:2014-01-01+ type:peexe itw:http fs:2016-10-30- submissions:3+ itw:http',
    #             outfile='../result_url_white_2.csv', init_offset=None)
    hashes = open('../result_url_white_2.csv').read().splitlines()[:20]
    res = conn.get_report(hashes[0], all_info=1, verbose=True, nb_process=1)
    print(res)
    # json.dump(res, open('temp2.json', 'w'))
    # urls = [hash_['ITW_urls'] for hash_ in res]
    # all_urls = [url for hash_ in urls for url in hash_]
    # open('../list_white_url.csv', 'w').write('\n'.join(all_urls))
    # json.dump(res, open('../example.json', 'w'))