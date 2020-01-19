from boto.s3.connection import S3Connection

conn = S3Connection('AKIAVGE22C6RZKKWACTB','JUDozTvjEAjfK7cg42gc8HE6rItQoYLU61kzfP3w')
bucket = conn.get_bucket('1polar')
for key in bucket.list():
    print(key.name)