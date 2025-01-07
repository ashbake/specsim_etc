# edits for meridian version
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from flask_sqlalchemy import SQLAlchemy
import io
import base64

# make plot fonts big
font = {'size'   : 14}
matplotlib.rc('font', **font)

# Load path to package and load a few useful things
print(os.getcwd())

sys.path.append('./templates/')
from flask import Flask, render_template, request, jsonify, Response, session

import numpy as np
import configparser
from datetime import datetime
import time
from celery import Celery

BASE_DIR = "/data/abaker/specsim_etc/user_data/" # location of user_data on meridian
DATA_DIR = '/data/abaker/data/'                  # location of specsim data on meridian

app = Flask(__name__, template_folder='./templates/')
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

celery = make_celery(app)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{BASE_DIR}/data.db'  # SQLite DB path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
app.secret_key = "super secret key"
db = SQLAlchemy(app)

# install specsim
sys.path.append('/data/abaker/specsim/')
from specsim.objects import load_object
from specsim.load_inputs import fill_data
from specsim.functions import *

# Data Model
class ComputedData(db.Model):
    # DATA MODEL for data storage
    # This class is used to store data from runs and query it out later
    # populate it with exact data will be downloading
    id = db.Column(db.Integer, primary_key=True)
    function_type = db.Column(db.String(20))
    x_values = db.Column(db.PickleType)  # Storing numpy array
    y_values = db.Column(db.PickleType)  
    snr_x    = db.Column(db.PickleType)
    snr_y    = db.Column(db.PickleType)
    rv_x     = db.Column(db.PickleType)
    rv_y     = db.Column(db.PickleType)
    thrpt_x  = db.Column(db.PickleType)
    thrpt_y  = db.Column(db.PickleType)
    ccf_vals = db.Column(db.PickleType) # length 4 for each band
    instrument = db.Column(db.String(30))
    configfile= db.Column(db.String(30))

# Create the table
with app.app_context():
    db.create_all()

def delete_old_cfg_files():
    current_time = time.time()
    ONE_WEEK =  7 * 24 * 60 * 60 #in seconds
    for file_name in os.listdir(BASE_DIR):
        if file_name.endswith(".cfg"):
            file_path = os.path.join(BASE_DIR, file_name)
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > ONE_WEEK:
                os.remove(file_path)

def check_and_clear_db():
    try:
        if os.path.getsize(BASE_DIR + 'data.db') > 5 * (1024**3):  # 5GB
            with app.app_context():
                ComputedData.query.delete()
                db.session.commit()
                # Vacuum the database to free up space
                db.engine.execute("VACUUM")
    except Exception as e:
        print(f"Error in check_and_clear_db: {e}")

def define_config_file(data):
    """
    Defines the config file based on data and run mode

    inputs
    -----
    data - user input data
    """
    instrument = data['instrument'] # pull outo instrument

    # some prep work on user data
    if data['zenith_angle']=='30':
        airmass = '1.2'
    elif data['zenith_angle']=='45':
        airmass= '1.4'
    elif data['zenith_angle']=='60':
        airmass='2'
    elif data['zenith_angle']=='0':
        airmass='1'

    # fix star temp to nearest 100 or 200 depending on temp based on what phoenix files we have downloaded
    if float(data['star_temperature']) <= 7400:
        star_temp = str(round(float(data['star_temperature'])/ 100) * 100)
    else:
        star_temp = str(round(float(data['star_temperature'])/ 200) * 200)

    # fill up config
    config = configparser.ConfigParser()
    config['run']={'plot_prefix':'testrun','savename':'test.txt','instrument':instrument,
                'mode':data['run_mode']}
    config['stel']={'phoenix_folder':DATA_DIR + '/phoenix/Z-0.0/','sonora_folder':DATA_DIR + 'sonora/',
                    'vsini':'0','rv':'0','teff':star_temp,'mag':data['star_magnitude'],'pl_rv':'0'}
    config['filt']={'zp_file':DATA_DIR + '/filters/zeropoints.txt','filter_path':DATA_DIR + 'filters/'
                    ,'band':data['filter'][-1],'family':data['filter'][:-2]}
    config['tel']={'telluric_file':DATA_DIR + 'telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits'
                ,'skypath':DATA_DIR + 'sky/','airmass':airmass,'pwv':data['pwv'],'seeing_set':data['atmospheric_conditions'],
                'zenith':data['zenith_angle']}
    config['inst']={'transmission_path': DATA_DIR + 'instrument/%s/throughput/' %instrument,
                    'order_bounds_file': DATA_DIR + 'instrument/%s/order_bounds.csv'%instrument,
                    'atm':'0',
                    'adc':'0',
                    'l0':'500',
                    'l1':'2650',
                    'res':'100000',
                    'res_samp':'3',
                    'pix_vert':'3',
                    'extraction_frac':'0.925',
                    'tel_area':'655',
                    'tel_diam':'30',
                    'readnoise':'12',
                    'darknoise':'0.01',
                    'saturation':'100000',
                    'pl_on':'1',
                    'rv_floor':'0.5',
                    'post_processing':'10'}
    config['obs']={'texp_frame_set':'default',
                    'nsamp':'16'}
    config['ao']={'inst':instrument,
                'mode':data['ao_mode'],
                'tt_static':'0',
                'lo_wfe':'10',
                'defocus':'10',
                'mag':'default',
                'teff':'default'}
    config['track']={'band':'JHgap',
                    'fratio':'35',
                    'camera':'h2rg',
                    'transmission_file':DATA_DIR + 'instrument/%s/track/trackingcamera.csv'%instrument,
                    'texp':'1',
                    'field_r':'0'}
    #config['coron']={'mode':'off-axis',
    #                'p_law_dh':'-2.0'}
    #config['etc']={'ccf':'on',
    #                'ccfetc':'no',
    #                'cal':'0.01'}
    #config['etc']={'ccf':'no',
    #                    'ccfetc':'open',
    #                    'goal_ccf':data['goal_ccf'], 
    #                    'SN':data['target_snr'], # TODO finish etc debugging and hispec debug
    #                    'texp_frame':data['frame_exposure_time']}
    #config['rv']={'water_only':'False',
    #                'line_spacing':'None',
    ##                'peak_spacing':'2e4',
    #                'height':'0.055',
    #                'cutoff':'0.01',
    #                'velocity_cutoff':'10',
    #                'rv_floor':'0.5'}

    # individual things for instrument type easier to define individually
    if instrument =='hispec':
        config['ao']['ttdynamic_set'] = DATA_DIR +'instrument/%s/ao/TTWFE_HAKA_092823.csv'%instrument
        config['ao']['ho_wfe_set']    = DATA_DIR +'instrument/%s/ao/HOWFE_HAKA_092823.csv'%instrument
        config['ao']['lo_wfe']        ='30'
        config['ao']['defocus']       ='25'
        config['inst']['atm']         = '1'
        config['inst']['adc']         = '1'
        #config['coron']['nactuators'] = '30',
        #config['coron']['fiber_contrast_gain'] = '3.'
    elif instrument=='modhis':
        config['ao']['ttdynamic_set'] = DATA_DIR +'instrument/%s/ao/TTDYNAMIC_NFIRAOS_091123.csv'%instrument
        config['ao']['ho_wfe_set']    = DATA_DIR +'instrument/%s/ao/HOWFE_NFIRAOS_091123.csv'%instrument
        config['ao']['lo_wfe']        ='10'
        config['ao']['defocus']       ='10'
        config['ao']['contrast_profile_path'] = DATA_DIR + 'instrument/%s/ao/contrastcurves/'%instrument
        #config['coron']['nactuators'] ='58' #TODO incorporate this into code input
        #config['coron']['fiber_contrast_gain'] = '10.'

    # decide on AO star properties
    if data['ao_star'] =='custom':
        config['ao']['mag']  = data['ao_star_mag']
        config['ao']['teff'] = data['ao_star_teff']
        
    # if off axis, define planet stuff
    if data['run_mode'] == 'snr_off' or data['run_mode'] =='etc_off':
        if float(data['planet_temperature']) <= 7400:
            plan_temp = str(round(float(data['planet_temperature'])/ 100) * 100)
        else:
            plan_temp = str(round(float(data['planet_temperature'])/ 200) * 200)
        
        config['stel']['pl_vsini'] = data['planet_vsini']
        config['stel']['pl_teff']  = plan_temp
        config['stel']['pl_mag']   = data['planet_magnitude']
        config['stel']['pl_sep']   = data['ang_sep']
    else: # define extra stellar stuff
        # make sure planet separation is 0 if not in off axis mode (after i edit code back to my own)
        config['stel']['vsini'] = data['vsini'] # these are only defined for star for on axis case
        config['stel']['rv']    = data['rv']
    
    # exposure time or frame time depending on etc or snr mode
    if data['run_mode'] == 'snr_off' or data['run_mode']=='snr_on':
        config['obs']['texp'] = data['exposure_time'] # only fill in exopsure time if in snr mode
    else:
        config['obs']['texp_frame'] = data['frame_exposure_time']
        config['obs']['target_snr'] = data['target_snr']
        config['obs']['target_ccf_snr'] = data['goal_ccf'] # use same user input
        
    return config

##############
# RUN SPECSIM AND FILL DATA MODEL FUNCTIONS
@celery.task
def async_fill_data(data,session_id):
    # define instrument, load config based on run mode and data
    #instrument = 'modhis'
    config = define_config_file(data)

    cfg_file_path = os.path.join(BASE_DIR, f"{session_id}config.cfg")
    with open(cfg_file_path, 'w') as configfile:
        config.write(configfile)
    configfile = cfg_file_path # define our config file name and path

    # run specsim!
    so    = load_object(configfile)  
    so.ao.mode = data['ao_mode']
    cload = fill_data(so) 

    # clear database if too big
    check_and_clear_db()

    # store outputs into database
    with app.app_context():
        # snr_off or snr_on bc code works either way the same
        computed_data_snr = ComputedData(
            function_type='data'+session_id, 
            x_values=so.obs.v[so.obs.ind_filter], # will be deprecated soon
            y_values=so.obs.snr[so.obs.ind_filter],
            snr_x   = so.obs.v[so.obs.ind_filter],
            snr_y   = so.obs.snr[so.obs.ind_filter],
            rv_x    = so.inst.order_cens,
            rv_y    = so.obs.rv_order,
            thrpt_x = so.inst.xtransmit,
            thrpt_y = so.inst.ytransmit,
            ccf_vals= [so.obs.ccf_snr_y, so.obs.ccf_snr_J, so.obs.ccf_snr_H, so.obs.ccf_snr_K],
            configfile=configfile,
            instrument=data.instrument
        )
        
        # add data to database then commit.......pretty sure this could be done by saving data to file then opening those files later but eh 
        db.session.add(computed_data_snr) 
        db.session.commit()

    delete_old_cfg_files()

@celery.task
def etc_async_task(data,session_id):
    # define instrument, load config based on run mode and data
    config = define_config_file(data)

    cfg_file_path = os.path.join(BASE_DIR, f"{session_id}config.cfg")
    with open(cfg_file_path, 'w') as configfile:
        config.write(configfile)
    configfile = cfg_file_path # define our config file name and path

    # run specsim!
    so    = load_object(configfile)  
    so.ao.mode = data['ao_mode']
    cload = fill_data(so) 

    # clear database if too big
    check_and_clear_db()

    # only run ccf snr etc if in off axis mode
    if data['run_mode'] == 'etc_off':
        ccf_vals = [so.obs.etc_ccf_snr_y,so.obs.etc_ccf_snr_J, so.obs.etc_ccf_snr_H, so.obs.etc_ccf_snr_K]
    else:
        ccf_vals = [-99,-99,-99,-99]

    with app.app_context():
        if data['run_mode'].startswith('etc'):
            computed_data = ComputedData(
                function_type='etc'+session_id, 
                x_values=so.inst.order_cens, 
                y_values=so.obs.etc_order_max,
                ccf_vals=ccf_vals,
                configfile=configfile,
                instrument=data.instrument
            )

            db.session.add(computed_data)
            db.session.commit()

    delete_old_cfg_files()

##############

@app.route('/submit_data', methods=['POST'])
def submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json  
    # define session id
    session['id_1']=time_index[:15]+'3'+ data['run_mode']
    task = async_fill_data.apply_async(args=[data,session['id_1']])
    print(session['id_1'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/status/{}'.format(task.id)}

@app.route('/download_csv', methods=['POST'])
def download_csv():
    # Retrieve the most recent x and y values for the given function type from the database
    computed_data  = ComputedData.query.filter_by(function_type='data'+session['id_1']).order_by(ComputedData.id.desc()).first()

    # Convert data to lists
    x_rv   = np.array(computed_data.rv_x).flatten().tolist()
    y_rv   = np.array(computed_data.rv_y).flatten().tolist()
    x_snr  = np.array(computed_data.snr_x).flatten().tolist()
    y_snr  = np.array(computed_data.snr_y).flatten().tolist()
    #if session['id_1'][16:] =='snr_off':

    # Create CSV data
    csv_data = "wavelength(nm),snr,order_cen(nm),rv_vals(m/s)\n"
    for i in range(max(len(x_rv),  len(x_snr))):
        val_x_rv = x_rv[i] if i < len(x_rv) else 'N/A'
        val_y_rv = y_rv[i] if i < len(y_rv) else 'N/A'
        val_x_snr = x_snr[i] if i < len(x_snr) else 'N/A'
        val_y_snr = y_snr[i] if i < len(y_snr) else 'N/A'

        csv_data += "{},{},{},{}\n".format(val_x_snr,val_y_snr,val_x_rv, val_y_rv)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename={}.csv".format('snr_results')}
    )

@app.route('/get_plot', methods=['GET'])
def get_plot():
    # This function generates the plot for snr_on and snr_off
    # Fetch the latest data from the database
    data_out = ComputedData.query.filter_by(function_type='data'+session['id_1']).order_by(ComputedData.id.desc()).first()
    order_cens = data_out.rv_x # order cens
    dv_vals    = data_out.rv_y # rv order
    
    # setup plot
    col_table = plt.get_cmap('Spectral_r')  
    fig, axs = plt.subplots(2,figsize=(10,10),sharex=True)
    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

    # plot SNR
    axs[0].plot(data_out.snr_x,data_out.snr_y,zorder=200,label='SNR')
    axs[0].set_ylabel('SNR')
    max_y = np.nanmax(data_out.snr_y) * 1.1
    axs[0].set_ylim(0, max_y)
    axs[0].fill_between([980,1100],-1,max_y,facecolor='k',edgecolor='black',alpha=0.1)
    axs[0].text(20+980,0.7*max_y, 'y')
    axs[0].fill_between([1170,1327],-1,max_y,facecolor='k',edgecolor='black',alpha=0.1)
    axs[0].text(50+1170,0.7*max_y, 'J')
    axs[0].fill_between([1490,1780],-1,max_y,facecolor='k',edgecolor='black',alpha=0.1)
    axs[0].text(50+1490,0.7*max_y, 'H')
    axs[0].fill_between([1990,2460],-1,max_y,facecolor='k',edgecolor='black',alpha=0.1)
    axs[0].text(50+1990,0.7*max_y, 'K')
    axs[0].grid('True')
    
    # plot throughput on second axis
    ax2 = axs[0].twinx()
    ax2.plot(data_out.thrpt_x,data_out.thrpt_y,'k',alpha=0.5,zorder=-100,label='Total Throughput')
    ax2.set_ylabel('Total Throughput',fontsize=12)
    
    # plot RV, diff colors per order
    for i,lam_cen in enumerate(order_cens):
        wvl_norm = (lam_cen - 900.) / (2500. - 900.)
        axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
    
    max_rv_lim = 3*np.median(dv_vals[np.where(~np.isinf(dv_vals))])
    if np.isnan(max_rv_lim): max_rv_lim = 1	
    axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
    axs[1].fill_between([1450,2400],0,max_rv_lim,facecolor='gray',alpha=0.2)
    axs[1].fill_between([980,1330],0,max_rv_lim,facecolor='gray',alpha=0.2)
    axs[1].grid('True')
    axs[1].set_ylim(-0,max_rv_lim)
    axs[1].set_xlim(950,2400)

    # compute sub RV for text
    sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
    sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
    dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
    dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
    dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
    dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

    # Add text to plot
    axs[1].text(1050,max_rv_lim/2,'$\sigma_{yJ}$=%s m/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
    axs[1].text(1500,max_rv_lim/2,'$\sigma_{HK}$=%s m/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
    axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
    axs[1].set_xlabel('Wavelength [nm]')
    axs[0].legend(fontsize=8,loc=2)
    ax2.legend(fontsize=8,loc=1)
    if session['id_1'][16:]== 'snr_off':
        axs[0].set_title('Off Axis, SNR')
    elif session['id_1'][16:]== 'snr_on':
        axs[0].set_title('On Axis, SNR')
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64 and return as JSON
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return jsonify({'image': img_base64})

@app.route('/ccf_snr_get_number', methods=['GET'])
def ccf_snr_get_number():
    data_out = ComputedData.query.filter_by(function_type='data'+session['id_1']).order_by(ComputedData.id.desc()).first()
    return jsonify({"y_band_snr": round(data_out.ccf_vals[0],1), 
                    "j_band_snr": round(data_out.ccf_vals[1],1),
                    "h_band_snr": round(data_out.ccf_vals[2],1),
                    "k_band_snr": round(data_out.ccf_vals[3],1),
                    })

########### ETC START

@app.route('/etc_submit_data', methods=['POST'])
def etc_submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json
    session['id_2']=time_index[:15]+'4'+ data['run_mode']
    task = etc_async_task.apply_async(args=[data,session['id_2']])
    print(session['id_2'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/etc_status/{}'.format(task.id)}

@app.route('/etc_download_csv', methods=['POST'])
def etc_download_csv():
    #if session['id_2'][16:]== 'etc_off':

    csv_filename = 'etc_result' 

    # Retrieve the most recent x and y values for the given function type from the database
    computed_data = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()

    x = np.array(computed_data.x_values).flatten().tolist()
    y = np.array(computed_data.y_values).flatten().tolist()
    #x3 = np.array(computed_data.ccf_vals).flatten().tolist()

    # Convert data to CSV format
    # TODO add header with all user params
    csv_data = "wavelength(nm),exptime(s)_for_SNR\n"
    for i in range(max(len(x), len(y))):
        val_x = x[i] if i < len(x) else 'N/A'
        val_y = y[i] if i < len(y) else 'N/A'
        csv_data += "{},{}\n".format(val_x, val_y)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename={}.csv".format(csv_filename)}
    )

@app.route('/etc_get_plot', methods=['GET'])
def etc_get_plot():
    # Fetch the latest data from the database
    data = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()
    order_cens    = data.x_values
    etc_order_max = data.y_values
    
    fig, axs = plt.subplots(1,figsize=(10,5),sharex=True)
    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)
    axs.plot(order_cens,etc_order_max,'o',zorder=200,label='SNR')

    # label
    axs.set_ylabel('Seconds')
    axs.set_title('ETC Per Order')
    axs.grid('True')
    axs.set_xlim(950,2400)
    axs.set_xlabel('Wavelength [nm]')
    axs.set_yscale("log")

    # filter band fill
    y_max_val = np.nanmax(etc_order_max)
    axs.fill_between([980,1100],0,y_max_val*1.1,facecolor='k',edgecolor='black',alpha=0.1)
    axs.text(50+980,y_max_val*.8, 'y')
    axs.fill_between([1170,1327],0,y_max_val*1.1,facecolor='k',edgecolor='black',alpha=0.1)
    axs.text(70+1170,y_max_val*.8, 'J')
    axs.fill_between([1490,1780],0,y_max_val*1.1,facecolor='k',edgecolor='black',alpha=0.1)
    axs.text(90+1490,y_max_val*.8, 'H')
    axs.fill_between([1990,2460],0,y_max_val*1.1,facecolor='k',edgecolor='black',alpha=0.1)
    axs.text(120+1990,y_max_val*.8, 'K')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64 and return as JSON
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return jsonify({'image': img_base64})

@app.route('/etc_ccf_snr_get_number', methods=['GET'])
def etc_ccf_snr_get_number():
    data = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()
    return jsonify({"ccf_etc_y": str(np.round(data.ccf_vals[0],1))+ ' sec',
                    "ccf_etc_j": str(np.round(data.ccf_vals[1],1))+ ' sec',
                    "ccf_etc_h": str(np.round(data.ccf_vals[2],1))+ ' sec',
                    "ccf_etc_k": str(np.round(data.ccf_vals[3],1))+ ' sec'
                    })


##################################################
# URL CALLS
############## 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/etc')
def etc():
    return render_template('etc.html')

@app.route('/hispec_snr')
def hispec_snr():
    return render_template('hispec_snr.html')

@app.route('/hispec_etc')
def hispec_etc():
    return render_template('hispec_etc.html')

##############################
# TASK STATUS
# F12 
@app.route('/status/<task_id>')
def task_status(task_id):
    task = async_fill_data.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
    else:
        response = {
            'state': task.state,
            'status': 'Task failed',
        }
    return jsonify(response)

@app.route('/etc_status/<task_id>')
def etc_task_status(task_id):
    task = etc_async_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
    else:
        response = {
            'state': task.state,
            'status': 'Task failed',
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)
    #app.run(debug=True)
