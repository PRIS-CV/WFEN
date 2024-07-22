import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir0', type=str, default="/home/liwenjie/wenjieli/Face_SR/datasets/test_datasets/Helen50/HR/") # Helen50 CelebA1000
parser.add_argument('--dir1', type=str, default="/home/liwenjie/wenjieli/Face_SR/WFEN/results_helen/wfen/") # results_helen results_CelebA
parser.add_argument('--out', type=str, default="./lpips.txt")
parser.add_argument('--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
sum = 0.0
count = 0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
    
		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()
		count += 1
      
		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		sum = sum + dist01
    #print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
sum = sum/count
print("sum = %.4f"%(sum))

f.close()
