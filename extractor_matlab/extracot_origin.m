%% This code only extracts

% Basic params
L=2;
SIZE=32;
template_d1=SIZE;
template_d2=SIZE;
color=true;
ccc=3;
ycbcr=false;
sharpen=false;

% Load trained networks
load('dlnetEncoder32_9_40.mat') % default
dlnetDecoder=dlnetDecoder1;

dirs=['dir_001'; 'dir_002'; 'dir_003'; 'dir_004'; 'dir_005'; 'dir_006'; 'dir_007'; 'dir_008'; 'dir_009'; 'dir_010';...
    'dir_011'; 'dir_012'; 'dir_013'; 'dir_014'; 'dir_015'; 'dir_016'; 'dir_017'; 'dir_018'; 'dir_019'; 'dir_020'];

check_dir = 'dataset/dlnetEncoder32_9_40_alpha20/watermarked_lab/jpeg50/';

for d=1:20
    files=dir([check_dir, dirs(d,:), '/*.png']);
    for i = 1:numel(files)
        filename = files(i).name;
        image = imread(([files(i).folder,'\', filename]));
        image_d1 = size(image, 1);
        image_d2 = size(image, 2);

        try
            test=rgb2gray(image);
        catch
            test=im2gray(image);
        end

        %test=imsharpen(test, 'Amount',1);

        test_cells=mat2cell(test, repmat(SIZE, 1, image_d1/SIZE), repmat(SIZE, 1, image_d2/SIZE));

        len=(image_d1/SIZE)*(image_d2/SIZE);

        test_n=zeros(SIZE,SIZE,1,len);
        for j=1:len
            test_n(:,:,:,j)=test_cells{j};
        end
        TEST_N=dlarray(test_n, "SSCB");
        m_test=extractdata(forward(dlnetDecoder, TEST_N));
        m_test=m_test(:,:);
        size(m_test)

        fileID = [check_dir, 'extracted/',dirs(d,:),'/', erase(filename, '.png'), '.xls'];
        writematrix(m_test(:), fileID);
    end
end

%% For one image
L=2;
SIZE=32;
template_d1=SIZE;
template_d2=SIZE;
color=true;
ccc=3;
ycbcr=false;
sharpen=false;

load('dlnetEncoder32_9_40.mat') % default
dlnetDecoder=dlnetDecoder1;

base_path = fileparts(mfilename('fullpath'));

image_path = fullfile(base_path,'..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', ...
    'watermarked', 'dir_001', '100.png');
image = imread(image_path);

image_d1 = size(image, 1);
image_d2 = size(image, 2);

try
    test=rgb2gray(image);
catch
    test=im2gray(image);
end

%test=imsharpen(test, 'Amount',1);

test_cells=mat2cell(test, repmat(SIZE, 1, image_d1/SIZE), repmat(SIZE, 1, image_d2/SIZE));

len=(image_d1/SIZE)*(image_d2/SIZE);

test_n=zeros(SIZE,SIZE,1,len);
for j=1:len
    test_n(:,:,:,j)=test_cells{j};
end
TEST_N=dlarray(test_n, "SSCB");
m_test=extractdata(forward(dlnetDecoder, TEST_N));
m_test=m_test(:,:);
size(m_test)

writematrix(m_test(:), '..\FullNS\100.xls');