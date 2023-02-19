BASE_URL='https://msropendataset01.blob.core.windows.net/motioncapturewithactionsmocapact-u20220731/'
FILE_NAME='transfer.tar.gz'
KEY='?sv=2019-02-02&sr=c&sig=2KaKBvC%2F3j8aPXpExjqPJojWhxHRWc72gYjzfQCFofg%3D&st=2023-02-18T16%3A57%3A46Z&se=2023-03-20T17%3A02%3A46Z&sp=rl'
OUTPUTPATH='transfer.tar.gz'

COMPLETE_PATH=$BASE_URL$FILE_NAME$KEY
echo $COMPLETE_PATH
azcopy copy $BASE_URL$FILE_NAME$KEY $OUTPUTPATH --recursive=true