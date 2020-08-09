import http.client
import mimetypes


def get_current_board(move, gameId):
    conn = http.client.HTTPSConnection("www.notexponential.com")
    dataList = []

    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList.append('--' + boundary)
    dataList.append('Content-Disposition: form-data; name=teamId;')

    dataList.append('Content-Type: {}'.format('multipart/form-data'))
    dataList.append('')

    dataList.append("1198")
    dataList.append('--' + boundary)
    dataList.append('Content-Disposition: form-data; name=move;')

    dataList.append('Content-Type: {}'.format('multipart/form-data'))
    dataList.append(move)

    dataList.append("")
    dataList.append('--' + boundary)
    dataList.append('Content-Disposition: form-data; name=type;')

    dataList.append('Content-Type: {}'.format('multipart/form-data'))
    dataList.append('')

    dataList.append("move")
    dataList.append('--' + boundary)
    dataList.append('Content-Disposition: form-data; name=gameId;')

    dataList.append('Content-Type: {}'.format('multipart/form-data'))
    dataList.append('')

    dataList.append(gameId)
    dataList.append('--' + boundary + '--')
    dataList.append('')
    body = '\r\n'.join(dataList)
    payload = body
    headers = {
        'x-api-key': 'c390b1f5889a538eca88',
        'userID': '881',
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("GET", "/aip2pgaming/api/index.php?gameId=" + gameId + "&type=boardMap", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")
