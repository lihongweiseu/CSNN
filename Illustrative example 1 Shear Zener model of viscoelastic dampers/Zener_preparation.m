clc, clear; close all; %warning off;
format long
% create a BLWN signals (starting from zero) for training and tesing
rng('default'); % To guarantee same results for every running
s = rng;

Niir = 8;
dt=[0.01,0.002,0.05]'; % sampling time/period
fs=1./dt; % sampling frequency. signal frequency range is [0,fs/2]
tend=[100,100,100]'-dt;
N=tend./dt+1;
amp=[32,128,8];
fend=[4,8,2]';
tin=4; % mutiply a 2-s half-window to make the data start from zero
Ntin=tin./dt;

Gz1=4.79; Gz2=0.479; etaz=0.248; A=6000; h=10;
Ah=A/h/1000;
a0=(Gz1+Gz2)/etaz; b0=Gz1*Gz2/etaz*Ah; b1=Gz1*Ah;
C=b0-b1*a0;

data=cell(9,1);
data{1}.t=(0:N(1)-1)'*dt(1);
u=wgn(N(1),1,10*log10(amp(1)));
iir = designfilt('lowpassiir','FilterOrder',Niir,'HalfPowerFrequency',fend(1),'SampleRate',fs(1));
windows=ones(N(1),1);
temp=hann(Ntin(1));
windows(1:Ntin(1)/2)=temp(1:Ntin(1)/2);
data{1}.u=filtfilt(iir,u).*windows;
data{1}.y=lsim(ss(-a0,1,C,b1),data{1}.u,data{1}.t,0);
data{1}.f=fs(1)*(0:N(1)/2)'/N(1);
u_fft=fft(data{1}.u,N(1));
P = abs(u_fft(1:N(1)/2+1));
P(2:end-1) = 2*P(2:end-1);
data{1}.P=P;
data{1}.N=N(1);

for i=1:2
    for j=1:2
        for k=1:2
            ii=4*i+2*j+k-5;
            data{ii}.t=(0:N(i+1)-1)'*dt(i+1);
            u=wgn(N(i+1),1,10*log10(amp(j+1)));
            iir = designfilt('lowpassiir','FilterOrder',Niir,'HalfPowerFrequency',fend(k+1),'SampleRate',fs(i+1));
            windows=ones(N(i+1),1);
            temp=hann(Ntin(i+1));
            windows(1:Ntin(i+1)/2)=temp(1:Ntin(i+1)/2);
            data{ii}.u=filtfilt(iir,u).*windows;
            data{ii}.y=lsim(ss(-a0,1,C,b1),data{ii}.u,data{ii}.t,0);
            data{ii}.f=fs(i+1)*(0:N(i+1)/2)'/N(i+1);
            u_fft=fft(data{ii}.u,N(i+1));
            P = abs(u_fft(1:N(i+1)/2+1));
            P(2:end-1) = 2*P(2:end-1);
            data{ii}.P=P;
            data{ii}.N=N(i+1);
        end
    end
end

%%
Train_data.t=data{1}.t';
Train_data.u=data{1}.u';
Train_data.y=data{1}.y';
Train_dt=dt(1);
Train_Nt=N(1);

Test_data=cell(2,1);
Test_data{1}.t=data{2}.t';
Test_data{1}.u=zeros(4,N(2));
Test_data{1}.y=zeros(4,N(2));
Test_data{2}.t=data{6}.t';
Test_data{2}.u=zeros(4,N(3));
Test_data{2}.y=zeros(4,N(3));
for i=1:2
    for j=1:4
        Test_data{i}.u(j,:)=data{4*i+j-3}.u';
        Test_data{i}.y(j,:)=data{4*i+j-3}.y';
    end
end
Test_dt=[dt(2);dt(3)];
Test_Nt=[N(2);N(3)];

save('Zener_data.mat','Train_data','Train_dt','Train_Nt','Test_data','Test_dt','Test_Nt'); %,'-double'