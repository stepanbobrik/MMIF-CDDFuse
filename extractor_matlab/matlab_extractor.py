import subprocess

def run_matlab_extractor(input_dir, output_dir):
    # MATLAB-команда с передачей путей
    cmd = f'matlab -batch "extractor(\'{input_dir}\', \'{output_dir}\')"'
    subprocess.run(cmd, shell=True)

def run_origin():
    subprocess.run('matlab -batch "run(\'C:/Users/user/MMIF-CDDFuse/extractor_matlab/extracot_origin.m\')"', shell=True)

if __name__ == '__main__':
    # run_matlab_extractor(
    #     input_dir="C:/FullNS/dlnetEncoder32_9_40_alpha20/watermarked",
    #     output_dir="C:/Users/user/MMIF-CDDFuse/FullNS/extracted_watermarks"
    # )
    run_origin()