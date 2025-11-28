inputFolder = "recordings"; 
outputFolder = "digit_dataset";

mkdir(outputFolder);

files = dir(fullfile(inputFolder, "*.wav"));

for i = 1:length(files)
    fname = files(i).name;
    parts = split(fname, "_");
    label = parts{1};      % leading digit

    outDir = fullfile(outputFolder, label);
    if ~exist(outDir, "dir")
        mkdir(outDir);
    end

    copyfile(fullfile(files(i).folder, fname), outDir);
end
