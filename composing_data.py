archivo = open('dataVL2.txt','w')

for i in range(0,100):
    for j in range(0,6):
        if len(str(i))==1:
            img1='AICDDataset\Images_NoShadow\Scene000'+str(i)+'_View0'+str(j)+'_'+'moving.png'
            img2='AICDDataset\Images_NoShadow\Scene000'+str(i)+'_View0'+str(j)+'_'+'target.png'
            mask='AICDDataset\GroundTruth\Scene000'+str(i)+'_View0'+str(j)+'_gtmask.png'        
        else:
            img1='AICDDataset\Images_NoShadow\Scene00'+str(i)+'_View0'+str(j)+'_'+'moving.png'
            img1='AICDDataset\Images_NoShadow\Scene00'+str(i)+'_View0'+str(j)+'_'+'target.png'
            mask='AICDDataset\GroundTruth\Scene00'+str(i)+'_View0'+str(j)+'_gtmask.png'
        archivo.write(img1+','+img2+','+mask+'\n')

archivo.close()