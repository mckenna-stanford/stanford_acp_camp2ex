import numpy as np
from pylab import *
import pdb


Rd   = 287.                         # Gas constant for dry air, J/(kg K)
Rv   = 461.6                        # Gas constant for water vapor, J/(kg K)
cp   = 7.*0.5*Rd                    # Specific heat of dry air at constant pressure, J/(kg K)
cv   = cp - Rd                      # Specific heat of dry air at constant volume, J/(kg K)
pref = 100000.0                     # reference sea level pressure, Pa
g    = 9.81                         # gravitational constant (m/s^2)


def vap_pres(r,p):
    # compute water vapor pressure using mixing ratio (r, in kg/kg) and pressure (p, in Pa):
    e = r*p/((Rd/Rv)+r)
    return e

def esat( t ): 
    # compute saturation vapor pressure based on temperature (in K). Output is in Pa.
    # WMO reference formula is that of Goff and Gratch (1946), slightly
    # modified by Goff in 1965:
    e1=101325.0
    TK=273.15
    esat = e1*np.power(10,(10.79586*(1-TK/t)-5.02808*np.log10(t/TK) + 1.50474*1e-4*(1-np.power(10,(-8.29692*(t/TK-1))))+0.42873*1e-3*(np.power(10,(4.76955*(1-TK/t)))-1)-2.2195983))
    return esat

def relhum( t, p, r ):
    # compute relative humidity based on temperature (t, in K), 
    # pressure (p in Pa) and mixing ratio (r, in kg/kg):
    rh = vap_pres(r,p)/esat(t)
    return rh

def Tsat( r ): 
    # compute saturation temperature from the mixing ratio (in kg/kg)
    r[np.where(r<1e-10000)] = np.nan
    r = r*1e3 # convert to g/kg
    tsat = 255.33 + 12.46*np.log(r*(1.0+.0265*r))
    return tsat

def theta_e( theta, r ):
    # from Bolton (1980), "The Computation of Equivalent Potential Temperature, MWR
    # compute equivalent potential temperature using potential temperature and mixing ratio:
    epott = theta*np.exp((3.376/Tsat(r)-.00254)*r*(1.0+.00081*r))
    #else:
    #   epott = np.nan
    return epott

def vort( u, v, dx, dy ):
    """
    Compute vorticity from u and v velocity components.
    
    Parameters:
    ----------
    u	: array, float
    	  2D field of u component of wind at grid centres.
    v	: array, float
    	  2D field of v component of wind at grid centres.
    dx	: 1D array or scalar, float
    	  delta-x values. If regular grid, dx can be a scalar.
    dy	: 1D array or scalar, float
    	  delta-y values. If regular grid, dy can be a scalar.
    
    Returns:
    -------
    vorticity: 2D array, float
    	   vorticity field.	
    """
    
    if u.shape != v.shape:
        print ('error computing vorticity: u and v fields must have the same dimensions!\n')
        return None
    else:
        if type(dx)!=ndarray:
            #if type(dx)==int or type(dx)==float:
            dx = np.ones(u.shape[0])*dx
        if type(dy)!=ndarray:
            #if type(dy)==int or type(dy)==float:
            dy = np.ones(v.shape[0])*dy
        nx,ny = u.shape
        vorticity = np.zeros([nx,ny])
        for ix in range(1,nx-1):
            for iy in range(1,ny-1):
                vorticity[ix,iy] = ((v[ix+1,iy] - v[ix-1,iy])/(2*dx[ix])) - ((u[ix,iy+1] - u[ix,iy-1])/(2*dy[iy]))
        vorticity[0,:]     = vorticity[1,:]
        vorticity[nx-1,:]  = vorticity[nx-2,:]
        vorticity[:,0]     = vorticity[:,1]
        vorticity[:,ny-1]  = vorticity[:,ny-2]
        
        return vorticity

def stream_function( u, v, dx, dy ):
    """
    Computes stream function from the velocity field.
    
    Paramters:
    ----------
    u	: 2D array, float.
    	  first component of wind field
    v	: 2D array, float.
    	  second component of wind field
    dx	: 1D array or scalar, float
    	  delta-x
    dy 	: 1D array or scalar, float
    	  delta-y
    Returns:
    --------
    psi	: 2D array, float
    	  stream function.
    """
    if u.shape != v.shape:
        print ('error computing stream function: u and v fields must have the same dimensions!\n')
        return None
    else:
        if type(dx)!=ndarray:
            #if type(dx)==int or type(dx)==float or type(dx)=float64:
            dx = np.ones(u.shape[0])*dx
        if type(dy)!=ndarray:
            #if type(dy)==int or type(dy)==float or type(dx)=float64:
            dy = np.ones(u.shape[1])*dy
        nx,ny = u.shape
    psi = np.zeros([nx,ny])
    for i in range(1,nx):
        psi[i,0] = psi[i-1,0] - dx[i]*0.5*(v[i-1,0]+v[i,0])
    for k in range(1,ny):
        psi[0,k] = psi[0,k-1] + dy[k]*0.5*(u[0,k-1]+u[0,k])
    corner = np.amin([nx,ny])
    for k in range(1,corner):
        for i in range(k,nx):
            storage_psi1 = psi[i-1,k] - dx[i]*0.5*(v[i-1,k]+v[i,k])
            storage_psi2 = psi[i,k-1] + dy[k-1]*0.5*(u[i,k-1]+u[i,k])
            psi[i,k] = 0.5*(storage_psi1 + storage_psi2)
        for j in range(k,ny):
            storage_psi3 = psi[k-1,j] - dx[i]*0.5*(v[k-1,j]+v[k,j])
            storage_psi4 = psi[k,j-1] + dy[k-1]*0.5*(u[k,j-1]+u[k,j])
            psi[k,j] = 0.5*(storage_psi3 + storage_psi4)
    return psi

def plot_mixing( angles, mixing, x, y, z, U, V, W, fname=None, xmin=None, xmax=None, ymin=None, ymax=None, scale=1e3, title=None, vmax=None, N=100, xlabel='X (R)', ylabel='Y (R)', zlabel='Z (R)', cblabel=None ):
    #xy plane:
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    speed_max = np.amax(np.sqrt(U*U+V*V+W*W))
    mixing=mixing*scale
    n_phi = np.where(angles[:,1]==angles[0,1])[0][1]
    n_alpha = len(angles)/n_phi
    cut = np.where(angles[:,1]==0)[0]  # take phi=0, or the mid point between -pi/2 and pi/2.
    if len(cut)==0:
        cut = n_phi/2
        angles_xy = angles[cut::n_phi,0]*180./np.pi
        data_xy = mixing[cut::n_phi]
    else:
        angles_xy = angles[cut,0]*180./np.pi
        data_xy = mixing[cut]
    #xz plane:
    cut1 = np.where(angles[:,0]==0)[0]      # take alpha(longitude) = 0 (or first angle if zero is not there)
    if len(cut1)==0:
        cut1 = np.where(angles[:,0]==angles[0,0])[0]
    cut2 = np.where(angles[:,0]==np.pi)[0]  # take alpha = pi/2 (or just the opposite from the first one)
    if len(cut2)==0:
        cut2 = np.where(angles[:,0] == angles[np.int(n_alpha/2)*n_phi,0])[0]
    cut = np.concatenate( (cut1,cut2) )
    angles_xz = np.zeros(len(cut))
    angles_xz[:len(cut1)] = angles[cut1,1]
    angles_xz[np.where(angles_xz <0)] =  angles_xz[np.where(angles_xz <0)]+2.*np.pi # get always positive angles
    dphi = np.abs(angles[1,1]-angles[0,1])
    angles_xz[len(cut1):] = np.arange(1.5*np.pi-(angles[cut2,1][0]+np.pi*0.5), np.pi*0.5, -dphi)
    #angles_xz[len(cut1):] = np.arange(angles[cut2,1][0]+2.*np.pi,np.pi*0.5,-dphi)
    data_xz = np.zeros(len(cut))
    data_xz[:len(cut1)] = mixing[cut1]
    data_xz[len(cut1):] = mixing[cut2]
    angles_xz = angles_xz*180./np.pi 
    #yz plane:
    cut1 = np.where(angles[:,0]==np.pi*0.5)[0]          # take alpha=pi/2 or the closest
    if len(cut1)==0:
        cut1 = np.where(angles[:,0]==angles[np.int(n_alpha/4.)*n_phi,0])[0]
    cut2 = np.where(angles[:,0]==1.5*np.pi)[0]          # take alpha=3pi/2 or the closest 
    if len(cut2)==0:
        cut2 = np.where(angles[:,0]==np.amin(angles[np.int(n_alpha*0.75)*n_phi,0]))[0]
    cut = np.concatenate( (cut1,cut2) )
    angles_yz = angles_xz
    data_yz = np.zeros(len(cut))
    data_yz[:len(cut1)] = mixing[cut1]
    data_yz[len(cut1):] = mixing[cut2]

    if vmax==None:
        vmax = np.nanmax(abs(mixing))
    else:
        vmax=vmax*scale
    jet = cm = create_colormap()
    cNorm  = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    fig=plt.figure(figsize=(20,6))
    #rc('font',**{'family':'FreeSans'})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    rc('text', usetex=False)
    if title !=None:
        fig.suptitle( title, fontsize=20, x=0.01, horizontalalignment='left' )
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.04, right=0.9, top=0.88, wspace=0.3)
    
    if title != None:
        plt.title(title, fontsize=20)

    ax1 = subplot(gs[0])
    plot_circle( angles_xy, data_xy, ax1, vmax, scalarMap )
    centre = np.where((z<1e-10)*(z>-1e-10))[0][0]
    crsection( x, y, U[:,:,centre].transpose(), V[:,:,centre].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=xlabel, ylabel=ylabel, title='XY plane', speed_max=speed_max )

    ax2 = subplot(gs[1])
    plot_circle( angles_xz, data_xz, ax2, vmax, scalarMap )
    centre = np.where((y<1e-10)*(y>-1e-10))[0][0]
    crsection( x, z, U[:,centre,:].transpose(), W[:,centre,:].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=xlabel, ylabel=zlabel, title='XZ plane', speed_max=speed_max )

    ax3 = subplot(gs[2])
    plot_circle( angles_yz, data_yz, ax3, vmax, scalarMap )
    centre = np.where((x<1e-10)*(x>-1e-10))[0][0]
    crsection( y, z, V[centre,:,:].transpose(), W[centre,:,:].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=ylabel, ylabel=zlabel, title='YZ plane', speed_max=speed_max )

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.94, right=0.96, top=0.88)
    ax4 = subplot(gs1[0])
    if scale!=1.:
        CB=matplotlib.colorbar.ColorbarBase(ax4, cmap=cm, norm=cNorm, orientation='vertical', format='%.1f', extend='max' )
        plt.title('      x %.E'%(1./scale), fontsize=16)
    else:
        CB=matplotlib.colorbar.ColorbarBase(ax4, cmap=cm, norm=cNorm, orientation='vertical')
    if cblabel!=None:
        CB.set_label(cblabel, fontsize=18, labelpad=-90, rotation=90, y=0.5)
    plt.yticks(fontsize=18)
    if fname==None:
        plt.show()
    else:
        for name in fname:
            plt.savefig( name )
    plt.close()
    plt.clf()


def plot_mixing_single( angles, mixing, x, y, z, U, V, W, fname=None, xmin=None, xmax=None, ymin=None, ymax=None, scale=1e3, title=None, vmax=None, N=100, xlabel='X (R)', ylabel='Y (R)', zlabel='Z (R)', cblabel=None, axis='xz' ):
    #xy plane:
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    speed_max = np.amax(np.sqrt(U*U+V*V+W*W))
    mixing=mixing*scale
    n_phi = np.where(angles[:,1]==angles[0,1])[0][1]
    n_alpha = len(angles)/n_phi
    cut = np.where(angles[:,1]==0)[0]  # take phi=0, or the mid point between -pi/2 and pi/2.
    if len(cut)==0:
        cut = n_phi/2
        angles_xy = angles[cut::n_phi,0]*180./np.pi
        data_xy = mixing[cut::n_phi]
    else:
        angles_xy = angles[cut,0]*180./np.pi
        data_xy = mixing[cut]
    #xz plane:
    cut1 = np.where(angles[:,0]==0)[0]      # take alpha(longitude) = 0 (or first angle if zero is not there)
    if len(cut1)==0:
        cut1 = np.where(angles[:,0]==angles[0,0])[0]
    cut2 = np.where(angles[:,0]==np.pi)[0]  # take alpha = pi/2 (or just the opposite from the first one)
    if len(cut2)==0:
        cut2 = np.where(angles[:,0] == angles[np.int(n_alpha/2)*n_phi,0])[0]
    cut = np.concatenate( (cut1,cut2) )
    angles_xz = np.zeros(len(cut))
    angles_xz[:len(cut1)] = angles[cut1,1]
    angles_xz[np.where(angles_xz <0)] =  angles_xz[np.where(angles_xz <0)]+2.*np.pi # get always positive angles
    dphi = np.abs(angles[1,1]-angles[0,1])
    angles_xz[len(cut1):] = np.arange(1.5*np.pi-(angles[cut2,1][0]+np.pi*0.5), np.pi*0.5, -dphi)
    #angles_xz[len(cut1):] = np.arange(angles[cut2,1][0]+2.*np.pi,np.pi*0.5,-dphi)
    data_xz = np.zeros(len(cut))
    data_xz[:len(cut1)] = mixing[cut1]
    data_xz[len(cut1):] = mixing[cut2]
    angles_xz = angles_xz*180./np.pi 
    #yz plane:
    cut1 = np.where(angles[:,0]==np.pi*0.5)[0]          # take alpha=pi/2 or the closest
    if len(cut1)==0:
        cut1 = np.where(angles[:,0]==angles[np.int(n_alpha/4.)*n_phi,0])[0]
    cut2 = np.where(angles[:,0]==1.5*np.pi)[0]          # take alpha=3pi/2 or the closest 
    if len(cut2)==0:
        cut2 = np.where(angles[:,0]==np.amin(angles[np.int(n_alpha*0.75)*n_phi,0]))[0]
    cut = np.concatenate( (cut1,cut2) )
    angles_yz = angles_xz
    data_yz = np.zeros(len(cut))
    data_yz[:len(cut1)] = mixing[cut1]
    data_yz[len(cut1):] = mixing[cut2]

    if vmax==None:
        vmax = np.nanmax(abs(mixing))
    else:
        vmax=vmax*scale
    jet = cm = create_colormap()
    cNorm  = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    fig=plt.figure(figsize=(7,6))
    #rc('font',**{'family':'FreeSans'})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    rc('text', usetex=False)
    if title !=None:
        fig.suptitle( title, fontsize=20, x=0.01, horizontalalignment='left' )
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.11, right=0.79, top=0.88, wspace=0.3)
    
    if title != None:
        plt.title(title, fontsize=20)

    ax1 = subplot(gs[0])
    if axis=='xy':
        plot_circle( angles_xy, data_xy, ax1, vmax, scalarMap )
        centre = np.where((z<1e-10)*(z>-1e-10))[0][0]
        crsection( x, y, U[:,:,centre].transpose(), V[:,:,centre].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=xlabel, ylabel=ylabel, title='XY plane', speed_max=speed_max )

    #ax2 = subplot(gs[1])
    if axis=='xz':
        plot_circle( angles_xz, data_xz, ax1, vmax, scalarMap )
        centre = np.where((y<1e-10)*(y>-1e-10))[0][0]
        crsection( x, z, U[:,centre,:].transpose(), W[:,centre,:].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=xlabel, ylabel=zlabel, title='XZ plane', speed_max=speed_max )

    #ax3 = subplot(gs[2])
    if axis=='yz':
        plot_circle( angles_yz, data_yz, ax1, vmax, scalarMap )
        centre = np.where((x<1e-10)*(x>-1e-10))[0][0]
        crsection( y, z, V[centre,:,:].transpose(), W[centre,:,:].transpose(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xlabel=ylabel, ylabel=zlabel, title='YZ plane', speed_max=speed_max )

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.86, right=0.92, top=0.88)
    ax4 = subplot(gs1[0])
    if scale!=1.:
        CB=matplotlib.colorbar.ColorbarBase(ax4, cmap=cm, norm=cNorm, orientation='vertical', format='%.1f', extend='max' )
        plt.title('      x %.E'%(1./scale), fontsize=16)
    else:
        CB=matplotlib.colorbar.ColorbarBase(ax4, cmap=cm, norm=cNorm, orientation='vertical')
    if cblabel!=None:
        CB.set_label(cblabel, fontsize=18, labelpad=-90, rotation=90, y=0.5)
    plt.yticks(fontsize=18)
    if fname==None:
        plt.show()
    else:
        for name in fname:
            plt.savefig( name )
    plt.close()
    plt.clf()

def plot_circle( angles, data, ax, vmax, scalarMap ):
    d_angle = angles[1]-angles[0]
    for i in range(len(angles)):
        arc = matplotlib.patches.Arc(xy=(0,0), width=2, height=2, theta1=angles[i]-d_angle*0.5, theta2=angles[i]+d_angle*0.5, linewidth=7, edgecolor=None, color=scalarMap.to_rgba(data[i]), zorder=2)
        ax.add_artist(arc)


def plot_field_streamlines( x_coor, y_coor, z_coor, field, U=None, V=None, W=None, vmin=None, vmax=None, R_coor=None, xmin=None, xmax=None, ymin=None, ymax=None, fname=None, centered_circle=None, zero_contour=False, xlabel='X (R)', ylabel='Y (R)', zlabel='Z (R)', cblabel=None, title=None, scale=1., symmetric=True, ticks_fmt='%.2f',x_contour=0 ):
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    # x_contour=X draws a contour at X, unless X=0 (does nothing)
    if vmax == None and vmin == None:
        vmax = np.around( np.nanmax(field),decimals=2 )
        vmin = np.around( np.nanmin(field),decimals=2 )
    elif vmax != None and vmin == None:
        vmax = np.around( vmax, decimals=2 )
        vmin = -vmax
    elif vmin != None and vmax != None:
        vmin = np.around( vmin, decimals=2 )
        vmax = np.around( vmax, decimals=2 )
    if scale != 1.:
        field = field*scale
        vmin=vmin*scale
        vmax=vmax*scale

    x2 = np.zeros(len(x_coor)+1)
    x2[:-1] = x_coor-(x_coor[1]-x_coor[0])*0.5
    x2[-1]=-x2[0]

    X, Y = np.meshgrid(x2, x2)
    data = np.ma.masked_array(field, mask=np.isnan(field))
    speed_max = np.amax(np.sqrt(U*U+V*V+W*W))
    fig=plt.figure(figsize=(20,6))
    #rc('font',**{'family':'FreeSans'})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    if title !=None:
        fig.suptitle( title, fontsize=20, x=0.01, horizontalalignment='left' )
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.04, right=0.9, top=0.88, wspace=0.3)
    
    ax1 = subplot(gs[0])
    centre = np.where((z_coor<1e-10)*(z_coor>-1e-10))[0][0]
    crsection( x_coor, y_coor, U[:,:,centre].transpose(), V[:,:,centre].transpose(), xmin, xmax, ymin, ymax, xlabel, ylabel, centered_circle, title='X-Y plane', speed_max=speed_max )
    if zero_contour:
        plt.contour(x_coor, y_coor, data[:,:,centre].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
    if x_contour!=0:
        plt.contour(x_coor, y_coor, data[:,:,centre].transpose(), 1, levels=[x_contour], linestyles='dashed',zorder=2)
    plt.pcolormesh( X, Y, data[:,:,centre].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')
    
    ax2 = subplot(gs[1])
    centre = np.where((y_coor<1e-10)*(y_coor>-1e-10))[0][0]
    crsection( x_coor, z_coor, U[:,centre,:].transpose(), W[:,centre,:].transpose(), xmin, xmax, ymin, ymax, xlabel, zlabel, centered_circle, title='X-Z plane', speed_max=speed_max )
    if zero_contour:
        plt.contour(x_coor, z_coor, data[:,centre,:].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
    if x_contour!=0:
        plt.contour(x_coor, y_coor, data[:,centre,:].transpose(), 1, levels=[x_contour], linestyles='dashed',zorder=2)
    plt.pcolormesh( X, Y, data[:,centre,:].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')
     
    ax3 = subplot(gs[2])
    centre = np.where((x_coor<1e-10)*(x_coor>-1e-10))[0][0]
    crsection( y_coor, z_coor, V[centre,:,:].transpose(), W[centre,:,:].transpose(), xmin, xmax, ymin, ymax, ylabel, zlabel, centered_circle, title='Y-Z plane', speed_max=speed_max )
    if zero_contour:
        plt.contour(y_coor, z_coor, data[centre,:,:].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
    if x_contour!=0:
        plt.contour(x_coor, y_coor, data[centre,:,:].transpose(), 1, levels=[x_contour], linestyles='dashed',zorder=2)
    plt.pcolormesh( X, Y, data[centre,:,:].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.94, right=0.96, top=0.88)
    ax4 = subplot(gs1[0])
    if symmetric:
        ticks=[vmin,vmin*0.5,0,vmax*0.5,vmax]
    else:
        ticks=[vmin,(vmin+vmax)*0.5,vmax]
    CB=plt.colorbar(ticks=ticks, format=ticks_fmt, cax=ax4 )
    plt.yticks(fontsize=16)
    #CB.ax.set_yticklabels(ticks, format='%.2f', fontsize=16)
    if cblabel!=None:
        if ticks_fmt=='%.2f':
            CB.set_label(cblabel, fontsize=18, rotation=90, y=0.5, labelpad=-98 )
        else:
            CB.set_label(cblabel, fontsize=18, rotation=90, y=0.5, labelpad=-70 )

    if scale!=1.:
        plt.title( 'x %.E'%(1./scale), y=1.02 )
    if fname == None:
        plt.show()
    else:
        for name in fname:
            plt.savefig(name)
    plt.close()
    plt.clf()

def plot_field_streamlines_single( x_coor, y_coor, z_coor, field, U=None, V=None, W=None, vmin=None, vmax=None, R_coor=None, xmin=None, xmax=None, ymin=None, ymax=None, fname=None, centered_circle=None, zero_contour=False, xlabel='X (R)', ylabel='Y (R)', zlabel='Z (R)', cblabel=None, title=None, scale=1., symmetric=True, ticks_fmt='%.1f', axis='xz',x_contour=0 ):
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    if vmax == None and vmin == None:
        vmax = np.around( np.nanmax(field),decimals=2 )
        vmin = np.around( np.nanmin(field),decimals=2 )
    elif vmax != None and vmin == None:
        vmax = np.around( vmax, decimals=2 )
        vmin = -vmax
    elif vmin != None and vmax != None:
        vmin = np.around( vmin, decimals=2 )
        vmax = np.around( vmax, decimals=2 )
    if scale != 1.:
        field = field*scale
        vmin=vmin*scale
        vmax=vmax*scale

    x2 = np.zeros(len(x_coor)+1)
    x2[:-1] = x_coor-(x_coor[1]-x_coor[0])*0.5
    x2[-1]=-x2[0]

    X, Y = np.meshgrid(x2, x2)
    data = np.ma.masked_array(field, mask=np.isnan(field))
    speed_max = np.amax(np.sqrt(U*U+V*V+W*W))
    fig=plt.figure(figsize=(7,6))
    #rc('font',**{'family':'FreeSans'})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    rc('text', usetex=False)
    #if title !=None:
    #    fig.suptitle( title, fontsize=20, x=0.01, horizontalalignment='left' )
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.11, right=0.79, top=0.88, wspace=0.3)
    
    ax1 = subplot(gs[0])
    if axis=='xy':
        centre = np.where((z_coor<1e-10)*(z_coor>-1e-10))[0][0]
        crsection( x_coor, y_coor, U[:,:,centre].transpose(), V[:,:,centre].transpose(), xmin, xmax, ymin, ymax, xlabel, ylabel, centered_circle, title=title, speed_max=speed_max )
        if zero_contour:
            plt.contour(x_coor, y_coor, data[:,:,centre].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
        if x_contour!=0:
            plt.contour(x_coor, y_coor, data[:,:,centre].transpose(), 1, levels=[x_contour], linestyles='dashed', zorder=2 )
        plt.pcolormesh( X, Y, data[:,:,centre].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')
    
    if axis=='xz':
        centre = np.where((y_coor<1e-10)*(y_coor>-1e-10))[0][0]
        crsection( x_coor, z_coor, U[:,centre,:].transpose(), W[:,centre,:].transpose(), xmin, xmax, ymin, ymax, xlabel, zlabel, centered_circle, title=title, speed_max=speed_max )
        if zero_contour:
            plt.contour(x_coor, z_coor, data[:,centre,:].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
        if x_contour!=0:
            plt.contour(x_coor, y_coor, data[:,centre,:].transpose(), 1, levels=[x_contour], linestyles='dashed', zorder=2 )
        plt.pcolormesh( X, Y, data[:,centre,:].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')
     
    if axis=='yz':
        centre = np.where((x_coor<1e-10)*(x_coor>-1e-10))[0][0]
        crsection( y_coor, z_coor, V[centre,:,:].transpose(), W[centre,:,:].transpose(), xmin, xmax, ymin, ymax, ylabel, zlabel, centered_circle, title=title, speed_max=speed_max )
        if zero_contour:
            plt.contour(y_coor, z_coor, data[centre,:,:].transpose(), 1, levels=[0], linestyles='dashed', zorder=2 )
        if x_contour!=0:
            plt.contour(x_coor, y_coor, data[centre,:,:].transpose(), 1, levels=[x_contour], linestyles='dashed', zorder=2 )
        plt.pcolormesh( X, Y, data[centre,:,:].transpose(), vmax=vmax, vmin=vmin, zorder=0, alpha=1, edgecolors='None')

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.82, right=0.88, top=0.88)
    ax4 = subplot(gs1[0])
    if symmetric:
        ticks=[vmin,vmin*0.5,0,vmax*0.5,vmax]
    else:
        ticks=[vmin,(vmin+vmax)*0.5,vmax]
    CB=plt.colorbar(ticks=ticks, format=ticks_fmt, cax=ax4 )
    plt.yticks(fontsize=16)
    #CB.ax.set_yticklabels(ticks, format='%.2f', fontsize=16)
    if cblabel!=None:
        CB.set_label(cblabel, fontsize=18, rotation=0, y=1.1, labelpad=-30 )
        #if ticks_fmt=='%.2f':
        #    CB.set_label(cblabel, fontsize=18, rotation=90, y=0.5, labelpad=-97 )
        #elif ticks_fmt=='%.1f':
        #    CB.set_label(cblabel, fontsize=18, rotation=90, y=0.5, labelpad=-87 )
        #else:
        #    CB.set_label(cblabel, fontsize=18, rotation=90, y=0.5, labelpad=-73 )

    if scale!=1.:
        plt.title( 'x %.E'%(1./scale), y=1.02 )
    if fname == None:
        plt.show()
    else:
        for name in fname:
            plt.savefig(name[:-4]+'_'+axis+name[-4:])
    plt.close()
    plt.clf()


def crsection( x, y, U, V, xmin, xmax, ymin, ymax, xlabel=None, ylabel=None, centered_circle=None, title=None, speed_max=None, streamlines_color='grey'):
    speed = np.sqrt( U*U+V*V )
    if speed_max==None:
        speed_max = speed.max()
    lw = 5*speed/speed_max
    plt.streamplot( x, y, U, V, linewidth=lw, color=streamlines_color )
    if xlabel!=None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel!=None:
        plt.ylabel(ylabel, fontsize=18)
    if xmin!=None and xmax!=None:
        plt.xlim(xmin,xmax)
    if ymin!=None and ymax!=None:
        plt.ylim(ymin,ymax)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if centered_circle!= None:
        ax = plt.gca()
        circle = Circle((0,0), centered_circle, facecolor='none', edgecolor='k', linewidth=3, zorder=1)
        ax.add_artist(circle)
    if title!=None:
        plt.title(title, fontsize=20)


def create_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {'red':  ((0.0, 0.0, 0.0),
                      (0.4, 0.5, 0.5),
                      (0.6, 1.0, 1.0),
                      (1.0,  1.0, 1.0)),

            'green': ((0.0, 0.0, 0.0),
                     (0.4, 0.0, 0.0),
                     (0.6, 0.0, 0.0),
                     (1.0,  1.0, 1.0)),

            'blue':  ((0.0, 0.0, 0.0),
                     (0.4, 0.0, 0.0),
                     (0.6, 0.0, 0.0),
                     (1.0,  0.0, 0.0))}
    return LinearSegmentedColormap('custom_colormap', cdict)


def plot_mixing_surface( angles, mixing ):
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.cos(angles[:,1])*np.cos(angles[:,0])
    Y = np.cos(angles[:,1])*np.sin(angles[:,0])
    Z = np.sin(angles[:,1])
    X, Y = np.meshgrid(X,Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_zlim3d(-2,2)
    ax.w_zaxis.set_major_locator(LinearLocator(6))
    plt.show()


