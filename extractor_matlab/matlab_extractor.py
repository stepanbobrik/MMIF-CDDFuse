import subprocess

def run_matlab_extractor(input_dir, output_dir):
    # MATLAB-команда с передачей путей
    print("batch extractor")
    cmd = f'matlab -batch "extractor(\'{input_dir}\', \'{output_dir}\')"'
    subprocess.run(cmd, shell=True)

def run_origin():
    subprocess.run('matlab -batch "run(\'C:/Users/user/MMIF-CDDFuse/extractor_matlab/extracot_origin.m\')"', shell=True)

if __name__ == '__main__':
    # run_matlab_extractor(
    #     input_dir="C:/Users/user/MMIF-CDDFuse/FullNS/dlnetEncoder32_9_40_alpha20/watermarked_lab/jpeg50/attacked_images",
    #     output_dir="C:/Users/user/MMIF-CDDFuse/FullNS/extracted_watermarks"
    # )
    run_matlab_extractor(
        input_dir="C:/Users/user/MMIF-CDDFuse/FullNS/resotred_mmif_cddfuse",
        output_dir="C:/Users/user/MMIF-CDDFuse/FullNS/mmif_cddfuse_extracted",
    )
    # run_origin()