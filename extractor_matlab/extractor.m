function extractor(input_root, output_root)
% input_root  — папка, где лежат dir_001 ... dir_020 с восстановленными изображениями
% output_root — куда сохранять .xls с извлечённым ЦВЗ (та же структура dir_xxx)

L = 2;
SIZE = 32;
template_d1 = SIZE;
template_d2 = SIZE;
color = true;
ccc = 3;
ycbcr = false;
sharpen = false;

% Загрузка модели один раз
load('dlnetEncoder32_9_40.mat') % default
dlnetDecoder = dlnetDecoder1;

% Папки dir_001 ... dir_020
dirs = cellstr(num2str((1:20)', 'dir_%03d'));

for d = 1:length(dirs)
    in_dir = fullfile(input_root, dirs{d});
    out_dir = fullfile(output_root, dirs{d});
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    files = dir(fullfile(in_dir, '*.png'));
    for i = 1:numel(files)
        filename = files(i).name;
        image = imread(fullfile(in_dir, filename));
        [image_d1, image_d2, ~] = size(image);

        try
            test = rgb2gray(image);
        catch
            test = im2gray(image);
        end

        test_cells = mat2cell(test, repmat(SIZE, 1, image_d1/SIZE), repmat(SIZE, 1, image_d2/SIZE));
        len = numel(test_cells);

        test_n = zeros(SIZE, SIZE, 1, len);
        for j = 1:len
            test_n(:, :, :, j) = test_cells{j};
        end

        TEST_N = dlarray(test_n, "SSCB");
        m_test = extractdata(forward(dlnetDecoder, TEST_N));
        m_test = m_test(:,:);

        out_path = fullfile(out_dir, [erase(filename, '.png'), '.xls']);
        writematrix(m_test(:), out_path);
    end
end
end
