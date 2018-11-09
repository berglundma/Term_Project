import os
import numpy as np
import csv
from scipy import stats
from scipy.integrate import quad
os.chdir('C:/users/emmac/documents/classes/6135/termproject')

#read the benchmark inputs
def read_input(file_name):
    f=open('%s.txt'%(file_name),'r')
    name=[]
    val=[]
    for line in f:
        t=line.split('\t')
        name.append(t[0])
        val.append(t[1].strip('\n'))
    L=float(val[0])
    Le=float(val[1])
    D=float(val[2])
    Pitch=float(val[3])
    Tm_in=float(val[4])
    mdot=float(val[5])
    Pnom=float(val[6])
    qpmax=float(val[7])
    Dci=float(val[8])
    Dfo=float(val[9])
    subcool=val[10]
    reactor=val[11]
    return [L,Le,D,Pitch,Tm_in,mdot,Pnom,qpmax,Dci,Dfo,subcool,reactor]
#the proptable.txt file had some white space at the end of the data, I deleted the white space, simplifying the read in process
#read properties table
def readprop(file_name="proptable"):
    g=open("%s.txt"%(file_name),'r')
    TC=[]
    Psat=[]
    volf=[]
    volg=[]
    hf=[]
    hg=[]
    muf=[]
    kf=[]
    Prf=[]
    mug=[]
    for line in g:
        t=line.split('\t')
        TC.append(float(t[0]))
        Psat.append(float(t[1]))
        volf.append(float(t[2]))
        volg.append(float(t[3]))
        hf.append(float(t[4]))
        hg.append(float(t[5]))
        muf.append(float(t[6]))
        kf.append(float(t[7]))
        Prf.append(float(t[8]))
        mug.append(float(t[9].strip('\n')))
    K=[TC,Psat,volf,volg,hf,hg,muf,kf,Prf,mug]
    return K
#write out solutions
def writeout(Tm,Tco,Tci,Tfo,Tfi,Tmax,x,xE,CHFR,dP,file_name):
    f=open('%s.txt'%(file_name),'w')
    f.write('Tm\tTco\tTci\tTfo\tTfi\tTmax\tx\txE\tCHFR\tdP\n')
    for i in xrange(len(Tm)):
        f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(Tm[i],Tco[i],Tci[i],Tfo[i],Tfi[i],Tmax[i],x[i],xE[i],CHFR[i],dP[i]))
    f.close

#returns htc using weismann    
def htc(inputs,properties):
    k=properties[7]
    Pr=np.array(properties[8])
    P=np.array(inputs[3])
    D=np.array(inputs[2])
    mdot=np.array(inputs[5])
    muf=np.array(properties[6])
    A=(D**2*np.pi/4.0)
    G=mdot/A
    Re=G*D/muf
    Nu_ct=1.826*(P/D)-1.0430
    phi=0.023*Re**0.8*Pr**.333
    Nu=Nu_ct*phi
    De=D*((4/np.pi)*(P/D)**2-1)
    return np.array(k*Nu/De)



#getting the control volume (CV) increments from the PWR,BWR files
def data_z(filename):
    # open the file in universal line ending mode 
    with open('%s.csv'%filename, 'rU') as infile:
    # read the file as a dictionary for each row ({header : value})
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
    z=data['z']
    return z

#interpolating the control volumes
def interpolate_CV(filename):
    
    Z=data_z(filename)
    z=[]
    for i in Z:
        z.append(float(i))
    cell=np.arange(1,len(z)+1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(cell,z)
    
    cc=np.linspace(1,len(z),2*len(z))
    return cc,cc*slope+intercept
    

#testing it
prop=readprop(file_name="proptable")
inputs=read_input(file_name='input')
gg=htc(inputs,prop)


#two phase HEM dp/dz calculation
#will likely need to use different two phase correlation and determine when the HEM is appropriate
def dpdz_HEM(inputs, properties,x):
    D=inputs[2]
    volf=np.array(prop[2])
    volg=np.array(prop[3])
    muf=np.array(prop[6])
    rhof=1/volf
    rhog=1/volg
    Psat=np.array(prop[1])
    mdot=inputs[5]
    A=np.pi*D**2/4.0
    G=mdot/A
    rhom=((x/rhog)+(1-x)/rhof)**-1
    volfg=1/rhog-1/rhof
    dpdz_grav=9.81*rhom
    
    Re_lo=G*D/muf
    f_lo=0.184*Re_lo**-.2 #reynolds number with mcadams correlation
    
    dxdz=np.diff(x)
    dpdz_acc= G**2*volfg*dxdz
    dpdz_fric=(f_lo/D)*(G**2/(2*rhom))
    
    return dpdz_fric+dpdz_acc+dpdz_grav
    
    
#Computing T_wall will be used later for the "marching inward"
def T_wall(Tbulk,qpp,htc):
    return Tbulk+qpp/htc


#this is the power function
def integrand(z, qo, Le):
    return qo*np.cos(np.pi*z/Le)
    


def T_m(z,inputs):
    L=inputs[0]
    Le=inputs[1]
    Tm_in=inputs[4]
    mdot=inputs[5]
    qo=inputs[7]
    
    cp=1
    
    
    I = Tm_in + (quad(integrand, -L/2, float(z), args=(qo,Le))[0])*(1/cp*mdot) 
    return I

def T_co(z,inputs, Tm):
    L=inputs[0]
    Le=inputs[1]
    Tm_in=inputs[4]
    mdot=inputs[5]
    qo=inputs[7]
    qp=integrand(z,qo,Le)
    Dfo=float(inputs[9])
    #need to get Rco
    Rco=Dfo/2.0
    term2=qp/(2*np.pi*Rco*htc)
    
    return Tm+term2
    
#11/5 current issues mainly that i do not know how to get c_p or what it is
