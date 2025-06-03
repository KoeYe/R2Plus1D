#!/bin/bash
mkdir data;
cd data;
mkdir something-something-v2;
cd something-something-v2;
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00;
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01;
wget -O labels.zip https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip;
cat 20bn-something-something-v2-?? >> 20bn-something-something-v2.tar.gz;
echo "Something-Something-V2 Complete!";
