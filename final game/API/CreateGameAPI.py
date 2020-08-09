import http.client
import mimetypes
conn = http.client.HTTPSConnection("www.notexponential.com")
dataList = []
boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=teamId1;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("1198")
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=teamId2;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("1197")
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=type;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("game")
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=gameType;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("TTT")
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=boardSize;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("12")
dataList.append('--' + boundary)
dataList.append('Content-Disposition: form-data; name=target;')

dataList.append('Content-Type: {}'.format('multipart/form-data'))
dataList.append('')

dataList.append("6")
dataList.append('--'+boundary+'--')
dataList.append('')
body = '\r\n'.join(dataList)
payload = body
headers = {
  'x-api-key': 'c390b1f5889a538eca88',
  'userID': '881',
  'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
}
conn.request("POST", "/aip2pgaming/api/index.php", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))