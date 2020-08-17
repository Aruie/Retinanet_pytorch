아옥ㅠㅠ

코코 다운
```
!wget http://images.cocodataset.org/zips/val2017.zip
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!mkdir images
!unzip -qq annotations_trainval2017.zip 
!unzip -qq val2017.zip -d images/
!unzip -qq train2017.zip -d images/
!rm *.zip
```