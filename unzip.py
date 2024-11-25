import zipfile
with zipfile.ZipFile("water.zip","r") as zip_ref:
    zip_ref.extractall("dataset")