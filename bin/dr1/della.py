'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def train_spenderq(study, tag, zmin, zmax, n_latent=10, similarity=True, consistency=True, debug=False):
    ''' deploy SpenderQ training with similiarty and consistency losses 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % study,
        ["#SBATCH --time=11:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/%s.o" % study, 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "", 
        "spender=/home/chhahn/projects/SpenderQ/bin/train_spender_updated.py",
        "dir_train=/tigress/chhahn/spender_qso/train/dr1", 
        "outfile=/tigress/chhahn/spender_qso/models/dr1/%s.pt" % study, 
        "python $spender $dir_train $outfile -t %s -n %i -zmin %f -zmax %f -l 100 %s %s" % (tag, n_latent, zmin, zmax, ['', '-s'][similarity], ['', '-c'][consistency]),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


def postprocess(model_name, input_tag, output_tag, ibatch0, ibatch1, sigma_lya=1.5, sigma_lyb=1.5, gpu=True, debug=True): 
    ''' postprocess SpenderQ outputs 
    '''
    # data direcotry
    dir_dat = '/tigress/chhahn/spender_qso/train/dr1'

    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_%s.%i_%i" % (input_tag, output_tag, ibatch0, ibatch1),
        "#SBATCH -o o/%s_%s.%i_%i.o" % (input_tag, output_tag, ibatch0, ibatch1), 
        ["#SBATCH --time=01:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', "#SBATCH --constraint=gpu80"][gpu], 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "postprocess=/home/chhahn/projects/SpenderQ/bin/postprocess.py", 
        "for ibatch in {%i..%i}; do python $postprocess %s /tigress/chhahn/spender_qso/models/dr1/%s.pt -ti %s -to %s -i $ibatch --sigma_lya %f --sigma_lyb %f; done" % (ibatch0, ibatch1, dir_dat, model_name, input_tag, output_tag, sigma_lya, sigma_lyb),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


def absorbers(absorber, model_tag, input_tag, ibatch, n_model=5, debug=True): 
    ''' find aborbers using the reconstructions 
    '''
    # data direcotry
    dir_dat = '/tigress/chhahn/spender_qso/absorbers'

    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_%s.%i" % (absorber, input_tag, ibatch),
        "#SBATCH -o o/%s_%s.%i.o" % (absorber, input_tag, ibatch),
        ["#SBATCH --time=01:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "find_absorbers=/home/chhahn/projects/SpenderQ/bin/find_absorbers.py", 
        "python $find_absorbers %s %s -m %s -nm %i -ti %s -i %i" % (dir_dat, absorber, model_tag, n_model, input_tag, ibatch),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


if __name__=="__main__": 
    #train_spenderq('qso.dr1.hiz', 'DESI.dr1.qsohiz', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first spenderq training with sim and con loss
    #postprocess('qso.dr1.hiz', 'DESI.dr1.qsohiz', 'DESI.dr1.qsohiz.i0', 0, 100, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.dr1.hiz.i0', 'DESI.dr1.qsohiz.i0', 2.1, 3.5, similarity=True, consistency=True, debug=False) # first iter
    #postprocess('qso.dr1.hiz.i0', 'DESI.dr1.qsohiz.i0', 'DESI.dr1.qsohiz.i1', 0, 100, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.dr1.hiz.i1', 'DESI.dr1.qsohiz.i1', 2.1, 3.5, similarity=True, consistency=True, debug=False) # second iter
    #postprocess('qso.dr1.hiz.i1', 'DESI.dr1.qsohiz.i1', 'DESI.dr1.qsohiz.i2', 0, 100, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    #train_spenderq('qso.dr1.hiz.i2', 'DESI.dr1.qsohiz.i2', 2.1, 3.5, similarity=True, consistency=True, debug=False) # third iter
    #postprocess('qso.dr1.hiz.i2', 'DESI.dr1.qsohiz.i2', 'DESI.dr1.qsohiz.i3', 0, 100, sigma_lya=1.1, sigma_lyb=0.8, gpu=True, debug=True)
    train_spenderq('qso.dr1.hiz.i3', 'DESI.dr1.qsohiz.i3', 2.1, 3.5, similarity=True, consistency=True, debug=False) # four iter
