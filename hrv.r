options(echo=FALSE)
library(RHRV)
#library(stringr)
options(digits.secs=3)

# Parameters ------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
session <- args[1]
participant <- args[2]
#session <- "matin"
#participant <- "p01"

BaseDir <- sprintf("/Users/olivier/Documents/ecole/maitrise/rawdata/%s/%s/bioharness", participant, session)
outputFilename <- sprintf("/Users/olivier/Documents/ecole/maitrise/rawdata/%s/%s/hrv.csv", participant, session)

BioFiles <- list.files(BaseDir)
BioRtoR <- file.path(BaseDir, BioFiles[grep(glob2rx("RR_local.csv"), BioFiles)])
#BioRtoR<- file.path(BaseDir, sprintf("%s-%s_RR.csv", date_str, time_str))

BioSummary <- file.path(BaseDir, BioFiles[grep(glob2rx("*Summary.csv"), BioFiles)])
summary <- read.csv(BioSummary, header=TRUE, sep=',', dec='.', fill=FALSE, skip=0)
startdate<- as.character(summary$Time[1])
starttime <- strptime(startdate, "%d/%m/%Y %H:%M:%OS")


RR<-read.csv(BioRtoR, header = TRUE, sep = ",", dec = ".", fill = FALSE,skip=0)

RR$Time[1]<-as.numeric(starttime)
for(t in 2:nrow(RR)){
  tminusone<-(t-1)
  previous_interval <- RR$Time[tminusone]
  current_RtoR <- (RR$RtoR[t]/1000.0)
  RR$Time[t]<-previous_interval+current_RtoR
}

RR$Time<-RR$Time*1000
# Création du data.frame à partir des données finales
RRClean<-cbind(RR$Time,RR$RtoR)
RRClean<-as.data.frame(RRClean)

s<-c("SensorHubTimestampInMillisecond", "RtoR")
names(RRClean)<-s
rm(s)
RRClean$RRcumsum<-cumsum(RRClean$RtoR)/1000

beats<-RRClean$RRcumsum
beats<-na.remove(beats)

hrv.data=CreateHRVData()
hrv.data = SetVerbose(hrv.data, TRUE )

hrv.data$Beat$Time<-beats
hrv.data$datetime<-as.POSIXlt(RRClean$SensorHubTimestampInMillisecond[1]/1000, origin="1970-01-01")

hrv.data=BuildNIHR(hrv.data)
hrv.data = FilterNIHR(hrv.data)
PlotNIHR(hrv.data)

hrv.data = InterpolateNIHR (hrv.data, freqhr = 4)
hrv.data = CreateFreqAnalysis(hrv.data)
hrv.data = CalculatePowerBand( hrv.data , indexFreqAnalysis= 1,size = 300, shift = 1, type = "fourier",ULFmin = 0, ULFmax = 0.03, VLFmin = 0.03, VLFmax = 0.05,LFmin = 0.05, LFmax = 0.15, HFmin = 0.15, HFmax = 0.4 )

SpectralHRV<-as.data.frame(hrv.data$FreqAnalysis)
SpectralHRV<-cbind(SpectralHRV$HRV,SpectralHRV$ULF,SpectralHRV$VLF,SpectralHRV$LF,SpectralHRV$HF,SpectralHRV$LFHF)
SpectralHRV<-as.data.frame(SpectralHRV)

names(SpectralHRV)<-c("HRV","ULF","VLF","LF","HF","LFHF")

start<-as.numeric(as.POSIXlt((RRClean$SensorHubTimestampInMillisecond[1]/1000), origin="1970-01-01"))

#SpectralHRV$SensorHubTimestampInMillisecond<-seq(from=start*1000,by=1000,along.with = SpectralHRV$HRV)

SpectralHRV$SensorHubTimestampInMillisecond<-as.numeric(as.POSIXlt((RRClean$SensorHubTimestampInMillisecond[1]/1000)+300, origin="1970-01-01"))
SpectralHRV$SensorHubTimestampInMillisecond<-seq(from=start*1000,by=1000,along.with = SpectralHRV$HRV)

#SpectralHRV$SensorHubTimestampInMillisecond<-as.numeric(SpectralHRV$SensorHubTimestampInMillisecond)*1000
SpectralHRV$ts <- as.POSIXct(SpectralHRV$SensorHubTimestampInMillisecond/1000, origin="1970-01-01")

#write.csv(SpectralHRV, outputFilename)

HRinst<-as.data.frame(hrv.data$HR)
names(HRinst)<-c("HR")
HRinst$SensorHubTimestampInMillisecond<-HRinst$HR
HRinst$SensorHubTimestampInMillisecond<-seq(from=start*1000,by=250,along.with = HRinst$SensorHubTimestampInMillisecond)

