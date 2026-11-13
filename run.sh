#!/usr/bin/env bash

########## PACS ##########
# python main.py --algorithm promptonly --stage all --dataset PACS --device cuda:0
# python main.py --algorithm feddog --stage all --dataset PACS --gamma 1 --device cuda:0
# python main.py --algorithm feddog --granularity class --stage all --dataset PACS --gamma 1 --device cuda:0

########## OfficeHome ##########
# python main.py --algorithm promptonly --stage all --dataset OfficeHome --images-per-class 16 --device cuda:0
# python main.py --algorithm feddog --stage all --dataset OfficeHome --shots-per-class 8 --images-per-class 16 --gamma 1 --device cuda:0
# python main.py --algorithm feddog --granularity class --stage all --dataset OfficeHome --shots-per-class 8 --images-per-class 16 --gamma 1 --device cuda:0

########## DomainNet ##########
# python main.py --algorithm promptonly --stage all --dataset DomainNet --device cuda:0
# python main.py --algorithm feddog --stage all --dataset DomainNet --gamma 2 --device cuda:0
# python main.py --algorithm feddog --granularity class --stage all --dataset DomainNet --gamma 1 --device cuda:0
