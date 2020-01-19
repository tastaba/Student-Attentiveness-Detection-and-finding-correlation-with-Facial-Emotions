import boto3
import json
from boto.s3.connection import S3Connection

def detect_multipleimage_faces(bucketName, connection):
    bucket = connection.get_bucket(bucketName)
    resultRows= []
    for image in bucket.list():
        resultRows.append(detect_faces(image.name,bucketName))
    print('Here are the Emotions:')
    for r in resultRows:
        print(r)
    print(len(resultRows))
    writeToCsv(resultRows)

def detect_faces(photo, bucket):
    client = boto3.client('rekognition')
    response = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':photo}},Attributes=['ALL'])
    #print('Detected faces for ' + photo)
    csvList= []
    csvList.append(photo)
    for faceDetail in response['FaceDetails']:
        result = json.dumps(faceDetail['Emotions'], indent=4, sort_keys=True)
        # create a JSON object parser; the result is a key-value pair dictionary
        parsed = json.loads(result)
        emotionDict = dict()
        for line in parsed:
            emotionDict[line['Type']] = line['Confidence']
        emotionDictKeys = sorted(emotionDict.keys())
        for e in emotionDictKeys:
            csvList.append(emotionDict[e])
        return csvList

def writeToCsv(rowsList):
    import csv
    with open('awsEmotion.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(rowsList)
        csvFile.close()

def main():
    bucket = '1polar'
    conn = S3Connection('AKIAVGE22C6RZKKWACTB', 'JUDozTvjEAjfK7cg42gc8HE6rItQoYLU61kzfP3w')
    detect_multipleimage_faces(bucket, conn)

if __name__ == "__main__":
    main()