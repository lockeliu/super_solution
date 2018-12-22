


python main.py -m newnet -t 'dataset/DIV2K/' -v 'dataset/Eval/Set14/'  -s 1,2,3,4 -g 4 -p weight/newnet/newnet.pt -r 30 -b 16 -e 200 > log/newnet.log
python main.py -m mdsr -t 'dataset/DIV2K/' -v 'dataset/Eval/Set14/'  -s 1,2,3,4 -g 4 -p weight/mdsr/mdsr.pt -r 30 -b 16  -e 200 > log/mdsr.log
