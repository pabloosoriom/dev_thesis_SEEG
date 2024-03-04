function st_Header = f_GetTRCHeader2(pstr_File)

s_fileID            =  fopen(pstr_File);

s_Status        = fseek(s_fileID, 175, 'bof');
s_HeaderType    = fread(s_fileID, 1,'*uchar')';


if (~isempty(s_HeaderType) && s_HeaderType ~= 4) || s_Status ~= 0 %#ok<BDSCI>
    error('[f_GetTRCHeader] - ERROR: Incorrect Type TRC Header!')
    return %#ok<UNRCH>
end

s_Status    = fseek(s_fileID, 0, 'bof');
if s_Status == 0
    st_Header.Title    = fread(s_fileID, 32,'*char')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 32, 'bof');
if s_Status == 0
    st_Header.Lab    = fread(s_fileID, 32,'*char')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 128, 'bof');
if s_Status == 0
    st_Header.RecDate      = fread(s_fileID, 3,'char')';
    st_Header.RecDate(3)	= st_Header.RecDate(3) + 1900;
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 131, 'bof');
if s_Status == 0
    st_Header.RecTime	= fread(s_fileID, 3,'char')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 136, 'bof');
if s_Status == 0
    st_Header.AcqType	= fread(s_fileID, 1,'ushort')';
else
    error('error')
end

s_Status    = fseek(s_fileID, 138, 'bof');
if s_Status == 0
    st_Header.DataStart	= fread(s_fileID, 1,'ulong')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 142, 'bof');
if s_Status == 0
    st_Header.ChanNum	= fread(s_fileID, 1,'ushort')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 144, 'bof');
if s_Status == 0
    st_Header.Multiplex	= fread(s_fileID, 1,'ushort')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 138, 'bof');
if s_Status == 0
    hdr.Data_Start_Offset=fread(s_fileID,1,'uint32');
    hdr.Num_Chan=fread(s_fileID,1,'uint16');
    hdr.Multiplexer=fread(s_fileID,1,'uint16');
    st_Header.Rate_Min=fread(s_fileID,1,'uint16');
    hdr.Bytes=fread(s_fileID,1,'uint16');    
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 146, 'bof');
if s_Status == 0
    s_MinRate	= fread(s_fileID, 1,'ushort')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 148, 'bof');
if s_Status == 0
    st_Header.NumBytes	= fread(s_fileID, 1,'ushort')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

fseek(s_fileID,st_Header.DataStart,-1);
datbeg = ftell(s_fileID);
fseek(s_fileID,0,1);
datend = ftell(s_fileID);
st_Header.Num_Samples = (datend-datbeg)/(st_Header.NumBytes*st_Header.ChanNum);
if rem(st_Header.Num_Samples, 1)~=0
    ft_warning('rounding off the number of samples');
    st_Header.Num_Samples = floor(st_Header.Num_Samples);
end


s_Status    = fseek(s_fileID, 176, 'bof');

if s_Status == 0
    st_Code.Name	= fread(s_fileID, 8,'*char')';
    s_Status        = fseek(s_fileID, 176+8, 'bof'); %#ok<*NASGU>
    st_Code.StartOf	= fread(s_fileID, 1,'ulong')';
    s_Status        = fseek(s_fileID, 176+12, 'bof');
    st_Code.Lenght	= fread(s_fileID, 1,'ulong')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, 192, 'bof');

if s_Status == 0
    st_Elec.Name	= fread(s_fileID, 8,'*char')';
    s_Status        = fseek(s_fileID, 192+8, 'bof'); %#ok<*NASGU>
    st_Elec.StartOf	= fread(s_fileID, 1,'ulong')';
    s_Status        = fseek(s_fileID, 192+12, 'bof');
    st_Elec.Lenght	= fread(s_fileID, 1,'ulong')';
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

s_Status    = fseek(s_fileID, st_Code.StartOf, 'bof');

if s_Status == 0
    v_ElectCh	= fread(s_fileID, st_Code.Lenght,'ushort');
    s_Idx       = find(v_ElectCh==0,1,'first')-1;
    v_ElectCh   = v_ElectCh(1:s_Idx)+1;
else
    error('[f_GetTRCHeader] - ERROR: fseek error!')
end

v_ElectOffsetData   = st_Elec.StartOf:128:st_Elec.StartOf+st_Elec.Lenght-1;
v_ElectOffsetData   = v_ElectOffsetData(v_ElectCh);
v_ElectData         = cell(st_Header.ChanNum,1);
st_Header.Ch        = cell(st_Header.ChanNum,1);
s_CurrVal           = 0;

for s_CurrPos	= v_ElectOffsetData
    
    s_Status        = fseek(s_fileID, s_CurrPos, 'bof');
    s_ElecStatus    = fread(s_fileID, 1,'*uchar');
    
    if s_ElecStatus == 0
        error('[f_GetTRCHeader] - ERROR: fseek error!')
    else        
        s_CurrVal	= s_CurrVal + 1;
    end
    
    
    s_Status        = fseek(s_fileID, s_CurrPos+1, 'bof');
    st_Info.Type    = fread(s_fileID, 1,'*uchar');
        
    s_Status        = fseek(s_fileID, s_CurrPos+2, 'bof');
    str_InLabelPos  = fread(s_fileID, 6,'*char')';
    s_Status        = fseek(s_fileID, s_CurrPos+8, 'bof');
    str_InLabelNeg  = fread(s_fileID, 6,'*char')';        
    st_Info.Ch      = char(str_InLabelPos);
    st_Info.Label   = strcat(str_InLabelPos,'_',str_InLabelNeg);    
    
    s_Status            = fseek(s_fileID, s_CurrPos+14, 'bof');
    st_Conv.LogicMin    = fread(s_fileID, 1,'int32')';
    s_Status            = fseek(s_fileID, s_CurrPos+18, 'bof');
    st_Conv.LogicMax    = fread(s_fileID, 1,'int32')';
    s_Status            = fseek(s_fileID, s_CurrPos+22, 'bof');
    st_Conv.LogicGnd    = fread(s_fileID, 1,'int32')';
    s_Status            = fseek(s_fileID, s_CurrPos+26, 'bof');
    st_Conv.PhysicMin	= fread(s_fileID, 1,'int32')';
    s_Status            = fseek(s_fileID, s_CurrPos+30, 'bof');
    st_Conv.PhysicMax   = fread(s_fileID, 1,'int32')';    
    
    s_Status        = fseek(s_fileID, s_CurrPos+34, 'bof');
    s_Units         = fread(s_fileID, 1,'*uchar');
    
    switch s_Units
        case -1
            st_Info.Units   = 'nV';
        case 0
            st_Info.Units   = 'uV';
        case 1
            st_Info.Units   = 'mV';
        case 2
            st_Info.Units   = 'V';
        case 100
            st_Info.Units   = '%';
        case 101
            st_Info.Units   = 'bpm';
        case 102
            st_Info.Units   = 'Adim';
        otherwise
    error('[f_GetTRCHeader] - ERROR: Electrode units unknown!')
    end
    
    s_Status        = fseek(s_fileID, s_CurrPos+44, 'bof');
    s_SamplingCoef	= fread(s_fileID, 1,'ushort');
    
    st_Info.Sampling    = s_MinRate * s_SamplingCoef;
    st_Info.Conversion  = st_Conv;
    
    v_ElectData{s_CurrVal}  = st_Info;
    st_Header.Ch{s_CurrVal,1} = deblank(st_Info.Ch);
end
v_ElectData         = v_ElectData(1:s_CurrVal);
st_Header.ElectData	= v_ElectData;

s_Status = fclose(s_fileID);