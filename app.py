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
# goal is to make this take the github version!
sys.path.append('/data/abaker/specsim/')
from specsim.objects import load_object
from specsim.load_inputs import fill_data
from specsim.functions import *

from flask import Flask, render_template, request, jsonify, Response,send_from_directory, session

import numpy as np
import configparser
from datetime import datetime
import time
from celery import Celery

BASE_DIR = "/data/abaker/specsim_etc/user_data/" # location of user_data on meridian
DATA_DIR = '/data/abaker/data/'                  # location of specsim data on meridian

app = Flask(__name__)
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

class ComputedData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    function_type = db.Column(db.String(20))
    x_values = db.Column(db.PickleType)  # Storing numpy array
    y_values = db.Column(db.PickleType)  # Storing numpy array

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
        if os.path.getsize('/scr3/specsim/user_data/data.db') > 5 * (1024**3):  # 5GB
            with app.app_context():
                ComputedData.query.delete()
                db.session.commit()
                # Vacuum the database to free up space
                db.engine.execute("VACUUM")
    except Exception as e:
        print(f"Error in check_and_clear_db: {e}")

def define_config_file(data,instrument):
    """
    Defines the config file based on data and run mode

    inputs
    -----
    data - user input data
    instrument - which instrument, modhis or hispec
    """
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
                ,'skypath':DATA_DIR + 'sky/','airmass':airmass,'pwv':data['pwv'],'seeing':data['atmospheric_conditions'],
                'zenith':data['zenith_angle']}
    config['inst']={'transmission_path':DATA_DIR + 'instrument/%s/throughput/' %instrument,
                    'order_bounds_file' : DATA_DIR + 'instrument/%s/order_bounds.csv'%instrument,
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
                'mag':'default'}
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
    #                    'SN':data['target_snr'],
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
        config['ao']['ttdynamic_set'] = DATA_DIR +'instrument/%s/ao/TT_dynamic.csv'%instrument
        config['ao']['ho_wfe_set']    = DATA_DIR +'instrument/%s/ao/HOwfe.csv'%instrument
        config['ao']['lo_wfe']        ='30'
        config['ao']['defocus']       ='25'
        #config['coron']['nactuators'] = '30',
        #config['coron']['fiber_contrast_gain'] = '3.'
    elif instrument=='modhis':
        config['ao']['ttdynamic_set'] = DATA_DIR +'instrument/%s/ao/TTDYNAMIC_NFIRAOS_091123.csv'%instrument
        config['ao']['ho_wfe_set']    = DATA_DIR +'instrument/%s/ao/HOWFE_NFIRAOS_091123.csv'%instrument
        config['ao']['lo_wfe']        ='10'
        config['ao']['defocus']       ='10'
        #config['coron']['nactuators'] ='58'
        #config['coron']['fiber_contrast_gain'] = '10.'

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

    return config

##############
@celery.task
def async_fill_data(data,session_id):
    # define instrument, load config based on run mode and data
    instrument = 'modhis'
    config = define_config_file(data, instrument)

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
        if data['run_mode'] == 'snr_off':
            computed_data = ComputedData(
                function_type='snr'+session_id, 
                x_values=so.obs.v[so.obs.ind_filter], 
                y_values=so.obs.p_snr
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit )
            computed_data3 = ComputedData(
                function_type='rv'+session_id, 
                x_values=so.rv.dv_spec_p, 
                y_values=so.rv.dv_tot_p.tolist()
            )
            computed_data4 = ComputedData(
                function_type='ccf'+session_id, 
                x_values=so.obs.ccf_snr.value, 
                y_values=so.obs.ccf_snr.value
            )
            computed_data5 = ComputedData(
                function_type='plot'+session_id, 
                x_values=so.rv.order_cen_lam, 
                y_values=so.rv.dv_vals_p )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data3)
            db.session.add(computed_data4)
            db.session.add(computed_data5)

            db.session.commit()
        if data['run_mode'] == 'snr_on':
            computed_data = ComputedData(
                function_type='snr'+session_id, 
                x_values=so.obs.v[so.obs.ind_filter], 
                y_values=so.obs.snr
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit )
            computed_data3 = ComputedData(
                function_type='rv'+session_id, 
                x_values=so.rv.dv_spec, 
                y_values=so.rv.dv_tot.tolist()
            )
            computed_data5 = ComputedData(
                function_type='plot'+session_id, 
                x_values=so.rv.order_cen_lam, 
                y_values=so.rv.dv_vals )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data3)
            db.session.add(computed_data5)

            db.session.commit()

    # rid of old config files while we're at it
    delete_old_cfg_files()

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = async_fill_data.AsyncResult(task_id)
    return jsonify({"status": str(task.status)})

##############
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_data', methods=['POST'])
def submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json  
    task = async_fill_data.apply_async(args=[data,time_index[:15]+'3'+data['run_mode']])
    session['id_1']=time_index[:15]+'3'+ data['run_mode']
    print(session['id_1'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/status/{}'.format(task.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
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

@app.route('/download_csv', methods=['POST'])
def download_csv():
    if session['id_1'][16:]== 'snr_off':
        function_type1 = session['id_1'][16:]
        
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data = ComputedData.query.filter_by(function_type='rv'+session['id_1']).order_by(ComputedData.id.desc()).first()
        computed_data3 = ComputedData.query.filter_by(function_type='plot'+session['id_1']).order_by(ComputedData.id.desc()).first()
        computed_data2 = ComputedData.query.filter_by(function_type='snr'+session['id_1']).order_by(ComputedData.id.desc()).first()
        computed_data5 = ComputedData.query.filter_by(function_type='ccf'+session['id_1']).order_by(ComputedData.id.desc()).first()

        # Convert data to lists
        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x3 = np.array(computed_data5.x_values).flatten().tolist()
        x4 = np.array(computed_data3.x_values).flatten().tolist()
        y4 = np.array(computed_data3.y_values).flatten().tolist()
        x5 = np.array(computed_data.x_values).flatten().tolist()
        y5 = np.array(computed_data.y_values).flatten().tolist()

        # Create CSV data
        csv_data = "wavelength(nm),snr,ccf,dv_spec,dv_total,order_cen,dv_vals\n"
        for i in range(max(len(x), len(y),len(x3), len(x4), len(y4), len(x5),len(y5))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x3 = x3[i] if i < len(x3) else 'N/A'
            val_x4 = x4[i] if i < len(x4) else 'N/A'
            val_y4 = y4[i] if i < len(y4) else 'N/A'
            val_x5 = x5[i] if i < len(x5) else 'N/A'
            val_y5 = y5[i] if i < len(y5) else 'N/A'
            
            csv_data += "{},{},{},{},{},{},{}\\n".format(val_x, val_y, val_x3, val_x4, val_y4, val_x5, val_y5)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )
    elif session['id_1'][16:]== 'snr_on':
        function_type1 = session['id_1'][16:]
        
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data = ComputedData.query.filter_by(function_type='rv'+session['id_1']).order_by(ComputedData.id.desc()).first()
        computed_data3 = ComputedData.query.filter_by(function_type='plot'+session['id_1']).order_by(ComputedData.id.desc()).first()
        computed_data2 = ComputedData.query.filter_by(function_type='snr'+session['id_1']).order_by(ComputedData.id.desc()).first()

        # Convert data to lists
        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x4 = np.array(computed_data3.x_values).flatten().tolist()
        y4 = np.array(computed_data3.y_values).flatten().tolist()
        x5 = np.array(computed_data.x_values).flatten().tolist()
        y5 = np.array(computed_data.y_values).flatten().tolist()

        # Create CSV data
        csv_data = "wavelength(nm),snr,dv_spec,dv_total,order_cen,dv_vals\n"
        for i in range(max(len(x), len(y), len(x4), len(y4), len(x5),len(y5))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x4 = x4[i] if i < len(x4) else 'N/A'
            val_y4 = y4[i] if i < len(y4) else 'N/A'
            val_x5 = x5[i] if i < len(x5) else 'N/A'
            val_y5 = y5[i] if i < len(y5) else 'N/A'
            
            csv_data += "{},{},{},{},{},{}\n".format(val_x, val_y, val_x4, val_y4, val_x5, val_y5)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )

@app.route('/get_plot', methods=['GET'])
def get_plot():
    # Fetch the latest data from the database
        if session['id_1'][16:]== 'snr_off':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='snr'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            data_entry = ComputedData.query.filter_by(function_type='plot'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values = data_entry.x_values
            y_values = data_entry.y_values
            order_cens=x_values
            dv_vals = y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(2,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
            axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
            axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
            axs[1].grid('True')
            axs[1].set_ylim(-0,3*np.median(dv_vals))
            axs[1].set_xlim(950,2400)
            axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
            axs[1].set_xlabel('Wavelength [nm]')

            axs[0].set_ylabel('SNR')
            axs[0].set_title('TMT-MODHIS, Off Axis')
            axs[0].fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(20+980,np.max(y_values_snr), 'y')
            axs[0].fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1170,np.max(y_values_snr), 'J')
            axs[0].fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1490,np.max(y_values_snr), 'H')
            axs[0].fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1990,np.max(y_values_snr), 'K')
            axs[0].grid('True')
            ax2 = axs[0].twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')

            ax2.set_ylabel('Total Throughput',fontsize=12)
            for i,lam_cen in enumerate(order_cens):
                wvl_norm = (lam_cen - 900.) / (2500. - 900.)
                axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
            axs[0].plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
            sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
            dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
            dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
            dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
            dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

            axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
            axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
            ax2.legend(fontsize=8,loc=1)
        elif session['id_1'][16:]== 'snr_on':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='snr'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            data_entry = ComputedData.query.filter_by(function_type='plot'+session['id_1']).order_by(ComputedData.id.desc()).first()
            x_values = data_entry.x_values
            y_values = data_entry.y_values
            order_cens=x_values
            dv_vals = y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(2,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
            axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
            axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
            axs[1].grid('True')
            axs[1].set_ylim(-0,3*np.median(dv_vals))
            axs[1].set_xlim(950,2400)
            axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
            axs[1].set_xlabel('Wavelength [nm]')

            axs[0].set_ylabel('SNR')
            axs[0].set_title('TMT-MODHIS, On Axis')
            axs[0].fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(20+980,np.max(y_values_snr), 'y')
            axs[0].fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1170,np.max(y_values_snr), 'J')
            axs[0].fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1490,np.max(y_values_snr), 'H')
            axs[0].fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1990,np.max(y_values_snr), 'K')
            axs[0].grid('True')
            ax2 = axs[0].twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            for i,lam_cen in enumerate(order_cens):
                wvl_norm = (lam_cen - 900.) / (2500. - 900.)
                axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
            y_values_snr = [x if x >= 0 else 0 for x in y_values_snr] 
            axs[0].plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
            sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
            dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
            dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
            dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
            dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

            axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
            axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
            ax2.legend(fontsize=8,loc=1)
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
    data_entry = ComputedData.query.filter_by(function_type='ccf'+session['id_1']).order_by(ComputedData.id.desc()).first()
    if data_entry:
        x_values = data_entry.x_values
        y_values = data_entry.y_values
    my_number =y_values
    return jsonify({"number": my_number})

###########
@celery.task
def new_async_task(data,session_id):
    # define instrument, load config based on run mode and data
    instrument = 'modhis'
    config = define_config_file(data, instrument)

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

    with app.app_context():
        if data['run_mode'] == 'etc_off':
            computed_data = ComputedData(
                function_type='etc'+session_id, 
                x_values=so.etc.v[so.obs.ind_filter], 
                y_values=so.etc.total_expt_filter
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit )
            computed_data4 = ComputedData(
                function_type='etc_ccf'+session_id, 
                x_values=so.obs.etc_ccf, 
                y_values=so.obs.etc_ccf
            )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data4)

            db.session.commit()
        if data['run_mode'] == 'etc_on':
            computed_data = ComputedData(
                function_type='etc'+session_id, 
                x_values=so.etc.v[so.obs.ind_filter], 
                y_values=so.etc.total_expt_s
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit )
            db.session.add(computed_data)
            db.session.add(computed_data2)

            db.session.commit()

    delete_old_cfg_files()

@app.route('/new_status/<task_id>')
def new_task_status(task_id):
    task = new_async_task.AsyncResult(task_id)
    return jsonify({"status": str(task.status)})

###########
@app.route('/etc')
def etc():
    return render_template('etc.html')

@app.route('/etc_snr_on_submit_data', methods=['POST'])
def etc_snr_on_submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json
    task = new_async_task.apply_async(args=[data,time_index[:15]+'4'+data['run_mode']])
    session['id_2']=time_index[:15]+'4'+ data['run_mode']
    print(session['id_2'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/new_status/{}'.format(task.id)}

@app.route('/new_status/<task_id>')
def newtaskstatus(task_id):
    task = new_async_task.AsyncResult(task_id)
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

@app.route('/etc_snr_on_download_csv', methods=['POST'])
def etc_snr_on_download_csv():
    if session['id_2'][16:]== 'etc_off':

        function_type1 = 'signal'

        # Retrieve the most recent x and y values for the given function type from the database
        computed_data2 = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()
        computed_data5 = ComputedData.query.filter_by(function_type='etc_ccf'+session['id_2']).order_by(ComputedData.id.desc()).first()

        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x3 = np.array(computed_data5.x_values).flatten().tolist()

        # Convert data to CSV format
        csv_data = "wavelength(nm),time(s)_for_SNR,time(s)_for_CCF\n"
        for i in range(max(len(x), len(y),len(x3))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x3 = x3[i] if i < len(x3) else 'N/A'
            csv_data += "{},{},{}\\n".format(val_x, val_y, val_x3)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )
    elif session['id_2'][16:]== 'etc_on':
        function_type1 = session['id_2'][16:]
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data2 = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()


        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()

        # Convert data to CSV format
        csv_data = "wavelength(nm),time(s)\n"
        for i in range(max(len(x), len(y))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            
            csv_data += "{},{}\\n".format(val_x, val_y)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )

@app.route('/etc_snr_on_get_plot2', methods=['GET'])
def etc_snr_on_get_plot2():
    # Fetch the latest data from the database
        if session['id_2'][16:]== 'etc_off':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_2']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(1,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs.set_ylabel('Seconds')
            axs.set_title('TMT-MODHIS, Off Axis')
            axs.grid('True')
            ax2 = axs.twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            axs.set_xlim(950,2400)
            axs.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Total Throughput',fontsize=12)
            axs.plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            axs.set_yscale("log")
            axs.fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(20+980,np.max(y_values_snr), 'y')
            axs.fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1170,np.max(y_values_snr), 'J')
            axs.fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1490,np.max(y_values_snr), 'H')
            axs.fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1990,np.max(y_values_snr), 'K')
            ax2.legend(fontsize=8,loc=1)
        elif session['id_2'][16:]== 'etc_on':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_2']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='etc'+session['id_2']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(1,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs.set_ylabel('Seconds')
            axs.set_title('TMT-MODHIS, On Axis')
            axs.grid('True')
            ax2 = axs.twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            axs.set_xlim(950,2400)
            axs.set_xlabel('Wavelength [nm]')
            y_values_snr = [x if x >= 0 else 0 for x in y_values_snr] 
            axs.plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            axs.set_yscale("log")# # 
            axs.fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(20+980,np.max(y_values_snr), 'y')
            axs.fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1170,np.max(y_values_snr), 'J')
            axs.fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1490,np.max(y_values_snr), 'H')
            axs.fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1990,np.max(y_values_snr), 'K')
            ax2.legend(fontsize=8,loc=1)
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
    data_entry = ComputedData.query.filter_by(function_type='etc_ccf'+session['id_2']).order_by(ComputedData.id.desc()).first()
    y_values = data_entry.y_values
    my_number =y_values
    return jsonify({"number": my_number})


##################################################
@celery.task
def hispec_async_fill_data(data,session_id):
    # define instrument, load config based on run mode and data
    instrument = 'hispec'
    config = define_config_file(data, instrument)

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

    with app.app_context():
        if data['run_mode'] == 'snr_off':
            computed_data = ComputedData(
                function_type='snr'+session_id, 
                x_values=so.obs.v[so.obs.ind_filter], 
                y_values=so.obs.p_snr
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit)
            computed_data3 = ComputedData(
                function_type='rv'+session_id, 
                x_values=so.rv.dv_spec_p, 
                y_values=so.rv.dv_tot_p.tolist()
            )
            computed_data4 = ComputedData(
                function_type='ccf'+session_id, 
                x_values=so.obs.ccf_snr.value, 
                y_values=so.obs.ccf_snr.value
            )
            computed_data5 = ComputedData(
                function_type='plot'+session_id, 
                x_values=so.rv.order_cen_lam, 
                y_values=so.rv.dv_vals_p )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data3)
            db.session.add(computed_data4)
            db.session.add(computed_data5)

            db.session.commit()
        if data['run_mode'] == 'snr_on':
            computed_data = ComputedData(
                function_type='snr'+session_id, 
                x_values=so.obs.v[so.obs.ind_filter], 
                y_values=so.obs.snr
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit)
            computed_data3 = ComputedData(
                function_type='rv'+session_id, 
                x_values=so.rv.dv_spec, 
                y_values=so.rv.dv_tot.tolist()
            )
            computed_data5 = ComputedData(
                function_type='plot'+session_id, 
                x_values=so.rv.order_cen_lam, 
                y_values=so.rv.dv_vals )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data3)
            db.session.add(computed_data5)

            db.session.commit()
    delete_old_cfg_files()

@app.route('/hispec_task/<task_id>', methods=['GET'])
def hispec_get_task_status(task_id):
    task = hispec_async_fill_data.AsyncResult(task_id)
    return jsonify({"status": str(task.status)})

##############
@app.route('/hispec_snr')
def hispec_snr():
    return render_template('hispec_snr.html')

@app.route('/hispec_submit_data', methods=['POST'])
def hispec_submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json  
    task = hispec_async_fill_data.apply_async(args=[data,time_index[:15]+'1'+data['run_mode']])
    session['id_4']=time_index[:15]+'1'+ data['run_mode']
    print(session['id_4'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/hispec_task/{}'.format(task.id)}

@app.route('/hispec_task/<task_id>')
def hispec_taskstatus(task_id):
    task = hispec_async_fill_data.AsyncResult(task_id)
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

@app.route('/hispec_download_csv', methods=['POST'])
def hispec_download_csv():
    if session['id_4'][16:]== 'snr_off':
        function_type1 = session['id_4'][16:]
        
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data = ComputedData.query.filter_by(function_type='rv'+session['id_4']).order_by(ComputedData.id.desc()).first()
        computed_data3 = ComputedData.query.filter_by(function_type='plot'+session['id_4']).order_by(ComputedData.id.desc()).first()
        computed_data2 = ComputedData.query.filter_by(function_type='snr'+session['id_4']).order_by(ComputedData.id.desc()).first()
        computed_data5 = ComputedData.query.filter_by(function_type='ccf'+session['id_4']).order_by(ComputedData.id.desc()).first()

        # Convert data to lists
        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x3 = np.array(computed_data5.x_values).flatten().tolist()
        x4 = np.array(computed_data3.x_values).flatten().tolist()
        y4 = np.array(computed_data3.y_values).flatten().tolist()
        x5 = np.array(computed_data.x_values).flatten().tolist()
        y5 = np.array(computed_data.y_values).flatten().tolist()

        # Create CSV data
        csv_data = "wavelength(nm),snr,ccf,dv_spec,dv_total,order_cen,dv_vals\n"
        for i in range(max(len(x), len(y),len(x3), len(x4), len(y4), len(x5),len(y5))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x3 = x3[i] if i < len(x3) else 'N/A'
            val_x4 = x4[i] if i < len(x4) else 'N/A'
            val_y4 = y4[i] if i < len(y4) else 'N/A'
            val_x5 = x5[i] if i < len(x5) else 'N/A'
            val_y5 = y5[i] if i < len(y5) else 'N/A'
            
            csv_data += "{},{},{},{},{},{},{}\\n".format(val_x, val_y, val_x3, val_x4, val_y4, val_x5, val_y5)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )
    elif session['id_4'][16:]== 'snr_on':
        function_type1 = session['id_4'][16:]
        
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data = ComputedData.query.filter_by(function_type='rv'+session['id_4']).order_by(ComputedData.id.desc()).first()
        computed_data3 = ComputedData.query.filter_by(function_type='plot'+session['id_4']).order_by(ComputedData.id.desc()).first()
        computed_data2 = ComputedData.query.filter_by(function_type='snr'+session['id_4']).order_by(ComputedData.id.desc()).first()

        # Convert data to lists
        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x4 = np.array(computed_data3.x_values).flatten().tolist()
        y4 = np.array(computed_data3.y_values).flatten().tolist()
        x5 = np.array(computed_data.x_values).flatten().tolist()
        y5 = np.array(computed_data.y_values).flatten().tolist()

        # Create CSV data
        csv_data = "wavelength(nm),snr,dv_spec,dv_total,order_cen,dv_vals\n"
        for i in range(max(len(x), len(y), len(x4), len(y4), len(x5),len(y5))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x4 = x4[i] if i < len(x4) else 'N/A'
            val_y4 = y4[i] if i < len(y4) else 'N/A'
            val_x5 = x5[i] if i < len(x5) else 'N/A'
            val_y5 = y5[i] if i < len(y5) else 'N/A'
            
            csv_data += "{},{},{},{},{},{}\n".format(val_x, val_y, val_x4, val_y4, val_x5, val_y5)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )

@app.route('/hispec_get_plot', methods=['GET'])
def hispec_get_plot():
    # Fetch the latest data from the database
        if session['id_4'][16:]== 'snr_off':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='snr'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            data_entry = ComputedData.query.filter_by(function_type='plot'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values = data_entry.x_values
            y_values = data_entry.y_values
            order_cens=x_values
            dv_vals = y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(2,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
            axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
            axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
            axs[1].grid('True')
            axs[1].set_ylim(-0,3*np.median(dv_vals))
            axs[1].set_xlim(950,2400)
            axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
            axs[1].set_xlabel('Wavelength [nm]')

            axs[0].set_ylabel('SNR')
            axs[0].set_title('Keck-HISPEC, Off Axis')
            axs[0].fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(20+980,np.max(y_values_snr), 'y')
            axs[0].fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1170,np.max(y_values_snr), 'J')
            axs[0].fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1490,np.max(y_values_snr), 'H')
            axs[0].fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1990,np.max(y_values_snr), 'K')
            axs[0].grid('True')
            ax2 = axs[0].twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')

            ax2.set_ylabel('Total Throughput',fontsize=12)
            for i,lam_cen in enumerate(order_cens):
                wvl_norm = (lam_cen - 900.) / (2500. - 900.)
                axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
            axs[0].plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
            sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
            dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
            dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
            dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
            dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

            axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
            axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
            ax2.legend(fontsize=8,loc=1)
        elif session['id_4'][16:]== 'snr_on':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='snr'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            data_entry = ComputedData.query.filter_by(function_type='plot'+session['id_4']).order_by(ComputedData.id.desc()).first()
            x_values = data_entry.x_values
            y_values = data_entry.y_values
            order_cens=x_values
            dv_vals = y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(2,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
            axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
            axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
            axs[1].grid('True')
            axs[1].set_ylim(-0,3*np.median(dv_vals))
            axs[1].set_xlim(950,2400)
            axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
            axs[1].set_xlabel('Wavelength [nm]')

            axs[0].set_ylabel('SNR')
            axs[0].set_title('Keck-HISPEC, On Axis')
            axs[0].fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(20+980,np.max(y_values_snr), 'y')
            axs[0].fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1170,np.max(y_values_snr), 'J')
            axs[0].fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1490,np.max(y_values_snr), 'H')
            axs[0].fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs[0].text(50+1990,np.max(y_values_snr), 'K')
            axs[0].grid('True')
            ax2 = axs[0].twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            for i,lam_cen in enumerate(order_cens):
                wvl_norm = (lam_cen - 900.) / (2500. - 900.)
                axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
            y_values_snr = [x if x >= 0 else 0 for x in y_values_snr] 
            axs[0].plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
            sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
            dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
            dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
            dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
            dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

            axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
            axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
            ax2.legend(fontsize=8,loc=1)
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode the image to base64 and return as JSON
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return jsonify({'image': img_base64})


@app.route('/hispec_ccf_snr_get_number', methods=['GET'])
def hispec_ccf_snr_get_number():
    data_entry = ComputedData.query.filter_by(function_type='ccf'+session['id_4']).order_by(ComputedData.id.desc()).first()
    if data_entry:
        x_values = data_entry.x_values
        y_values = data_entry.y_values
    my_number =y_values
    return jsonify({"number": my_number})


###########
@celery.task
def hispec_new_async_task(data,session_id):
    instrument = 'hispec'
    config = define_config_file(data, instrument)

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

    with app.app_context():
        if data['run_mode'] == 'etc_off':
            computed_data = ComputedData(
                function_type='etc'+session_id, 
                x_values=so.etc.v[so.obs.ind_filter], 
                y_values=so.etc.total_expt_filter
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit)
            computed_data4 = ComputedData(
                function_type='etc_ccf'+session_id, 
                x_values=so.obs.etc_ccf, 
                y_values=so.obs.etc_ccf
            )
            db.session.add(computed_data)
            db.session.add(computed_data2)
            db.session.add(computed_data4)

            db.session.commit()
        if data['run_mode'] == 'etc_on':
            computed_data = ComputedData(
                function_type='etc'+session_id, 
                x_values=so.etc.v[so.obs.ind_filter], 
                y_values=so.etc.total_expt_s
            )
            computed_data2 = ComputedData(
                function_type='sr'+session_id, 
                x_values=so.inst.xtransmit, 
                y_values=so.inst.ytransmit)
            db.session.add(computed_data)
            db.session.add(computed_data2)

            db.session.commit()
    delete_old_cfg_files()

@app.route('/hispec_new_status/<task_id>')
def hispec_new_task_status(task_id):
    task = hispec_new_async_task.AsyncResult(task_id)
    return jsonify({"status": str(task.status)})

###########
@app.route('/hispec_etc')
def hispec_etc():
    return render_template('hispec_etc.html')

@app.route('/hispec_etc_snr_on_submit_data', methods=['POST'])
def hispec_etc_snr_on_submit_data():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    time_index = str(formatted_time)
    data = request.json  
    task = hispec_new_async_task.apply_async(args=[data,time_index[:15]+'2'+data['run_mode']])
    session['id_3']=time_index[:15]+'2'+ data['run_mode']
    print(session['id_3'])
    # Process the received data as required
    # For now, just print it to the console
    print(data)
    return jsonify({}), 202, {'Location': '/hispec_new_status/{}'.format(task.id)}

@app.route('/hispec_new_status/<task_id>')
def hispec_newtaskstatus(task_id):
    task = hispec_new_async_task.AsyncResult(task_id)
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

@app.route('/hispec_etc_snr_on_download_csv', methods=['POST'])
def hispec_etc_snr_on_download_csv():
    if session['id_3'][16:]== 'etc_off':

        function_type1 = 'signal'

        # Retrieve the most recent x and y values for the given function type from the database
        computed_data2 = ComputedData.query.filter_by(function_type='etc'+session['id_3']).order_by(ComputedData.id.desc()).first()
        computed_data5 = ComputedData.query.filter_by(function_type='etc_ccf'+session['id_3']).order_by(ComputedData.id.desc()).first()


        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()
        x3 = np.array(computed_data5.x_values).flatten().tolist()

        # Convert data to CSV format
        csv_data = "wavelength(nm),time(s)_for_SNR,time(s)_for_CCF\n"
        for i in range(max(len(x), len(y),len(x3))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            val_x3 = x3[i] if i < len(x3) else 'N/A'
            csv_data += "{},{},{}\\n".format(val_x, val_y, val_x3)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )
    elif session['id_3'][16:]== 'etc_on':
        function_type1 = session['id_3'][16:]
        # Retrieve the most recent x and y values for the given function type from the database
        computed_data2 = ComputedData.query.filter_by(function_type='etc'+session['id_3']).order_by(ComputedData.id.desc()).first()


        x = np.array(computed_data2.x_values).flatten().tolist()
        y = np.array(computed_data2.y_values).flatten().tolist()

        # Convert data to CSV format
        csv_data = "wavelength(nm),time(s)\n"
        for i in range(max(len(x), len(y))):
            val_x = x[i] if i < len(x) else 'N/A'
            val_y = y[i] if i < len(y) else 'N/A'
            
            csv_data += "{},{}\\n".format(val_x, val_y)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}.csv".format(function_type1)}
        )

@app.route('/hispec_etc_snr_on_get_plot2', methods=['GET'])
def hispec_etc_snr_on_get_plot2():
    # Fetch the latest data from the database
        if session['id_3'][16:]== 'etc_off':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_3']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='etc'+session['id_3']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(1,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs.set_ylabel('Seconds')
            axs.set_title('Keck-HISPEC, Off Axis')
            axs.grid('True')
            ax2 = axs.twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            axs.set_xlim(950,2400)
            axs.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Total Throughput',fontsize=12)
            axs.plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            axs.set_yscale("log")
            axs.fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(20+980,np.max(y_values_snr), 'y')
            axs.fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1170,np.max(y_values_snr), 'J')
            axs.fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1490,np.max(y_values_snr), 'H')
            axs.fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1990,np.max(y_values_snr), 'K')
            ax2.legend(fontsize=8,loc=1)
        elif session['id_3'][16:]== 'etc_on':
            data_entry_sr = ComputedData.query.filter_by(function_type='sr'+session['id_3']).order_by(ComputedData.id.desc()).first()
            x_values_sr = data_entry_sr.x_values
            y_values_sr = data_entry_sr.y_values
            data_entry_snr = ComputedData.query.filter_by(function_type='etc'+session['id_3']).order_by(ComputedData.id.desc()).first()
            x_values_snr = data_entry_snr.x_values
            y_values_snr = data_entry_snr.y_values
            col_table = plt.get_cmap('Spectral_r')
            fig, axs = plt.subplots(1,figsize=(10,10),sharex=True)
            plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.3,right=0.85,top=0.85)

            axs.set_ylabel('Seconds')
            axs.set_title('Keck-HISPEC, On Axis')
            axs.grid('True')
            ax2 = axs.twinx() 
            ax2.plot(x_values_sr,y_values_sr,'k',alpha=0.5,zorder=-100,label='Total Throughput')
            axs.set_xlim(950,2400)
            axs.set_xlabel('Wavelength [nm]')
            y_values_snr = [x if x >= 0 else 0 for x in y_values_snr] 
            axs.plot(x_values_snr,y_values_snr,zorder=200,label='SNR')
            axs.set_yscale("log")# # 
            axs.fill_between([980,1100],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(20+980,np.max(y_values_snr), 'y')
            axs.fill_between([1170,1327],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1170,np.max(y_values_snr), 'J')
            axs.fill_between([1490,1780],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1490,np.max(y_values_snr), 'H')
            axs.fill_between([1990,2460],0,np.max(y_values_snr),facecolor='k',edgecolor='black',alpha=0.1)
            axs.text(50+1990,np.max(y_values_snr), 'K')
            ax2.legend(fontsize=8,loc=1)
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode the image to base64 and return as JSON
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return jsonify({'image': img_base64})

@app.route('/hispec_etc_ccf_snr_get_number', methods=['GET'])
def hispec_etc_ccf_snr_get_number():
    data_entry = ComputedData.query.filter_by(function_type='etc_ccf'+session['id_3']).order_by(ComputedData.id.desc()).first()
    y_values = data_entry.y_values
    my_number =y_values
    return jsonify({"number": my_number})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)
    #app.run(debug=True)
