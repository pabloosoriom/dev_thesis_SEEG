filename = 'data/EEG_353.TRC';

st_Header = f_GetTRCHeader2(filename);
Fs = st_Header.Rate_Min;
Samples = st_Header.Num_Samples;
N_sec = Samples/(Fs);

% [hdr, datamatrix] = read_micromed_trc(filename)
time1 = 1415168; %Samples
time2 = 1568708; % Samples
data = read_micromed_trc(filename, time1, time2);
vtime = (time1:time2)/Fs;

% Graficar 1 canal
figure; plot(data(: ,:));
% Graficar todos los canales
data1 = data-mean(data,2);
m_offset=repmat((0:size(data1,1)-1)'*500,1,size(data1,2));
figure; plot(vtime,(data1+m_offset)');
set(gca,'YTick',(0:size(data1,1)-1)*500,'YTickLabels',string(st_Header.Ch));
