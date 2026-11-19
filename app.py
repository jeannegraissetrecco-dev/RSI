"""Interface web RSI — couche de présentation, indépendante du solveur.

Lancer avec `streamlit run app.py`. Compatible Streamlit Community Cloud.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

# Le serveur cloud n'a pas d'écran ; PyVista doit pouvoir utiliser OSMesa.
os.environ.setdefault('VTK_USE_X', 'OFF')
os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkOSOpenGLRenderWindow')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from physics import run_simulation

st.set_page_config(
    page_title='RSI | Thermique & mécanique du béton',
    page_icon='🧱',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('''<style>
.block-container { padding-top: 1.7rem; max-width: 1380px; }
[data-testid="stMetric"] { background: #f3f8fb; padding: 13px 17px; border-radius: 10px; border: 1px solid #e0e8ef; }
[data-testid="stSidebar"] { background: #f6f9fc; }
.page-kicker {color: #54778f; font-size: 0.83rem; letter-spacing: .12em; font-weight: 700; text-transform: uppercase;}
.hero-subtitle {color: #6c7d88; margin-top: -.7rem; margin-bottom: 1.3rem;}
</style>''', unsafe_allow_html=True)

FACES_2D = [('clxinf','Gauche (x−)'), ('clxsup','Droite (x+)'),
            ('clyinf','Bas (y−)'), ('clysup','Haut (y+)')]
FACES_3D = FACES_2D + [('clzinf','Bas (z−)'), ('clzsup','Haut (z+)')]
INITIAL_BC = ['Coffrage bois', 'Coffrage tôle', 'Pièce béton', 'Symétrie']
FINAL_BC = ['Bord libre', 'Pièce béton', 'Symétrie']

DEFAULTS = dict(clxinf='Coffrage bois', clxsup='Symétrie', clyinf='Symétrie', clysup='Symétrie',
                clzinf='Symétrie', clzsup='Symétrie', clxinf2='Bord libre',
                clxsup2='Symétrie', clyinf2='Symétrie', clysup2='Symétrie',
                clzinf2='Symétrie', clzsup2='Symétrie')


def parameters_form():
    """Toutes les valeurs sont converties dans les unités SI attendues par le moteur."""
    with st.sidebar:
        st.markdown('### Paramétrage de l’étude')
        dim = st.radio('Dimension', [2, 3], format_func=lambda d: f'{d}D', horizontal=True, key='dimension_select')
        with st.form('simulation_form'):
            with st.expander('1 · Géométrie et maillage', expanded=True):
                longueur = st.number_input('Longueur X (m)', min_value=0.01, value=1.0, step=0.25, format='%.3f')
                largeur = st.number_input('Largeur Y (m)', min_value=0.01, value=1.0, step=0.25, format='%.3f')
                hauteur = st.number_input('Hauteur Z (m)', min_value=0.01, value=1.0, step=0.25, format='%.3f') if dim==3 else 1.0
                default_step = .25 if dim==3 else .10
                cols = st.columns(2 if dim==2 else 3)
                with cols[0]: hx = st.number_input('Δx (m)', min_value=.001, value=default_step, step=.05, format='%.3f')
                with cols[1]: hy = st.number_input('Δy (m)', min_value=.001, value=default_step, step=.05, format='%.3f')
                if dim==3:
                    with cols[2]: hz = st.number_input('Δz (m)', min_value=.001, value=.25, step=.05, format='%.3f')
                else: hz=1.0
            with st.expander('2 · Béton et propriétés thermomécaniques'):
                lam = st.number_input('Conductivité λ (W/m/K)', min_value=.001, value=2.30, step=.1, format='%.3f')
                cp = st.number_input('Capacité thermique Cp (J/kg/K)', min_value=1., value=1000., step=50.)
                rho = st.number_input('Masse volumique (kg/m³)', min_value=1., value=2400., step=50.)
                nu = st.number_input('Coefficient de Poisson ν', min_value=0., max_value=.49, value=.30, step=.01)
                alpha = st.number_input('Dilatation thermique α (× 10⁻⁶/K)', min_value=0., value=10., step=.5)
                efin = st.number_input('Module d’Young final (GPa)', min_value=.01, value=30., step=1.)
                st.caption('La loi d’hydratation et le potentiel RSI restent ceux du script corrigé.')
            with st.expander('3 · Environnement et durée'):
                t0 = st.number_input('Température de coulage (°C)', min_value=-20., max_value=100., value=20., step=1.)
                text = st.number_input('Température extérieure (°C)', min_value=-50., max_value=100., value=20., step=1.)
                hair = st.number_input('Convection à l’air h (W/m²/K)', min_value=.001, value=10., step=1.)
                drc = st.number_input('Décoffrage à (h)', min_value=0., value=48., step=6.)
                duree = st.number_input('Durée maximale (jours)', min_value=.01, value=10., step=1.)
                deltatmin = st.number_input('Seuil de température d’arrêt (°C)', value=54., step=1.)
                st.caption('Arrêt du calcul selon la règle du code historique : durée maximale ou température inférieure au seuil après au moins 4 jours.')
            with st.expander('4 · Conditions aux faces', expanded=False):
                st.caption('Une condition de face agit sur la convection et sur les blocages mécaniques, conformément au modèle historique corrigé.')
                boundaries={}
                for key,label in (FACES_3D if dim==3 else FACES_2D):
                    st.markdown(f'**{label}**')
                    ca, cb = st.columns(2)
                    with ca:
                        d=DEFAULTS[key]; boundaries[key] = st.selectbox('Pendant prise', INITIAL_BC, index=INITIAL_BC.index(d),key=f'{key}_initial')
                    with cb:
                        d=DEFAULTS[key+'2']; boundaries[key+'2'] = st.selectbox('Après décoffrage', FINAL_BC, index=FINAL_BC.index(d),key=f'{key}_final')
                for key, _ in FACES_3D:
                    boundaries.setdefault(key,DEFAULTS[key]); boundaries.setdefault(key+'2',DEFAULTS[key+'2'])
            submitted = st.form_submit_button('▶ Lancer la simulation', type='primary', use_container_width=True)
    params=dict(dim=dim,lam=lam,C_p=cp,rho=rho,nu=nu,alpha=alpha*1e-6,Efin=efin*1e9,
                T0=t0+273.15,Text=text+273.15,duree=duree*86400,deltatmin=deltatmin,
                longueur=longueur,largeur=largeur,hauteur=hauteur,hx=hx,hy=hy,hz=hz,
                drc=drc*3600,hair=hair,**boundaries)
    return submitted, params


def graph_history(result, kind='temperature'):
    fig=go.Figure()
    if kind=='temperature':
        ts = result['times_h']
        for key, color, label in [('tmax_C','#ad2941','T max'),('tcentre_C','#1496b8','T au centre')]:
            fig.add_scatter(x=ts,y=result[key],mode='lines',name=label,line=dict(color=color,width=2.7))
        fig.update_layout(yaxis_title='Température (°C)',xaxis_title='Temps depuis le coulage (h)')
    elif kind=='beton':
        ts=result['times_h']
        fig.add_scatter(x=ts,y=result['young_GPa'],name='Module E (GPa)',line=dict(color='#15496f',width=2.5))
        fig.add_scatter(x=ts,y=result['resistance_MPa'],name='Résistance fcm (MPa)',yaxis='y2',line=dict(color='#d79140',width=2.5))
        fig.update_layout(yaxis=dict(title='E (GPa)'),yaxis2=dict(title='fcm (MPa)',overlaying='y',side='right'),xaxis_title='Temps (h)')
    elif kind=='epsilon':
        fig.add_scatter(x=result['times_epsilon_h'],y=result['epsilon_max_percent'],mode='lines',name='Max ε∞',line=dict(color='#8c3ca7',width=2.6))
        fig.update_layout(yaxis_title='Déformation RSI ultime potentielle (%)',xaxis_title='Temps jeune âge (h)')
    fig.update_layout(template='plotly_white',height=340,margin=dict(t=15,b=45,l=45,r=45),legend=dict(orientation='h',y=1.13),hovermode='x unified')
    if len(result['times_h']):
        drc=result['params']['drc']/3600
        if drc<=result['times_h'][-1]:fig.add_vline(x=drc,line_dash='dot',line_color='gray',annotation_text='Décoffrage')
    return fig


def chart_2d(field, result):
    a=field['data']; step_x,step_y=result['pas_m'][:2]
    if field['position']=='cell':
        xx=(np.arange(a.shape[1])+.5)*step_x; yy=(np.arange(a.shape[0])+.5)*step_y
    else:
        xx=np.arange(a.shape[1])*step_x; yy=np.arange(a.shape[0])*step_y
    divergent=field['unit'] in ('MPa','mm')
    bound=float(np.nanmax(np.abs(a))) if a.size else 0
    extra={'zmin':-bound,'zmax':bound} if divergent and bound>0 else {}
    fig=go.Figure(go.Heatmap(x=xx,y=yy,z=a,colorscale='RdBu_r' if divergent else 'Viridis',
                            colorbar=dict(title=field['unit']),hovertemplate='x=%{x:.3f} m<br>y=%{y:.3f} m<br>Valeur=%{z:.5g}<extra></extra>',**extra))
    fig.update_layout(template='plotly_white',height=530,margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title='x (m)',yaxis_title='y (m)',
                      yaxis=dict(scaleanchor='x',scaleratio=1))
    return fig


def create_pyvista_grid(result, field_name):
    """Champs z,y,x -> tableau VTK x-fast, éléments ou points selon le champ."""
    import pyvista as pv
    f=result['fields'][field_name]
    if result['dimension']!=3: raise ValueError('Maillage PyVista disponible uniquement en 3D.')
    nz,ny,nx = result['fields']['Température au pic thermique']['data'].shape
    grid=pv.ImageData(dimensions=(nx+1,ny+1,nz+1),spacing=tuple(result['pas_m']),origin=(0,0,0))
    name=f'{field_name} [{f["unit"]}]'
    values=np.asarray(f['data'],dtype=float).ravel(order='C')
    if f['position']=='cell':grid.cell_data[name]=values
    else:grid.point_data[name]=values
    return grid,name


def plot_3d_pyvista(result,field_name):
    import pyvista as pv
    from stpyvista import stpyvista
    grid,name=create_pyvista_grid(result,field_name)
    L,W,H=result['dimensions_m']; f=result['fields'][field_name]
    clim=None
    if f['unit'] in ('MPa','mm'):
        val=float(np.max(np.abs(f['data'])))
        if val>0:clim=[-val,val]
    plotter=pv.Plotter(off_screen=True,window_size=(920,600))
    plotter.set_background('#ffffff')
    slices=grid.slice_orthogonal(x=L*.5,y=W*.5,z=H*.5)
    plotter.add_mesh(slices,scalars=name,cmap='coolwarm' if clim else 'viridis',clim=clim,
                     show_edges=False,scalar_bar_args={'title':f['unit'],'vertical':True})
    plotter.add_mesh(grid.outline(), color='#536f83', line_width=2)
    plotter.add_axes()
    plotter.view_isometric()
    stpyvista(plotter,key=f'viewer-{field_name}')
    plotter.close()


def plot_3d_fallback(result,field_name):
    """Affichage 3D Plotly en secours si le moteur VTK sans écran échoue."""
    f=result['fields'][field_name]
    a=f['data']
    hx,hy,hz=result['pas_m']; z_n,y_n,x_n=a.shape
    offset=0.0 if f['position']=='point' else .5
    xx=(np.arange(x_n)+offset)*hx; yy=(np.arange(y_n)+offset)*hy; zz=(np.arange(z_n)+offset)*hz
    k=z_n//2;j=y_n//2;i=x_n//2
    cmin=float(np.min(a));cmax=float(np.max(a))
    if cmin==cmax:cmax=cmin+1.e-9
    diverging=f['unit'] in ('MPa','mm')
    if diverging:
        b=max(abs(cmin),abs(cmax));cmin,cmax=-b,b
        if b==0:cmin,cmax=-1e-9,1e-9
    fig=go.Figure()
    # Les trois plans orthogonaux correspondent à des coupes à mi-hauteur.
    X,Y=np.meshgrid(xx,yy)
    fig.add_trace(go.Surface(x=X,y=Y,z=np.full_like(X,zz[k]),surfacecolor=a[k,:,:],
                             colorscale='RdBu' if diverging else 'Viridis',cmin=cmin,cmax=cmax,showscale=True,colorbar=dict(title=f['unit'])))
    X,Z=np.meshgrid(xx,zz)
    fig.add_trace(go.Surface(x=X,y=np.full_like(X,yy[j]),z=Z,surfacecolor=a[:,j,:],
                             colorscale='RdBu' if diverging else 'Viridis',cmin=cmin,cmax=cmax,showscale=False))
    Y,Z=np.meshgrid(yy,zz)
    fig.add_trace(go.Surface(x=np.full_like(Y,xx[i]),y=Y,z=Z,surfacecolor=a[:,:,i],
                             colorscale='RdBu' if diverging else 'Viridis',cmin=cmin,cmax=cmax,showscale=False))
    fig.update_layout(template='plotly_white',height=600,margin=dict(l=0,r=0,b=0,t=0),
                      scene=dict(xaxis_title='x (m)',yaxis_title='y (m)',zaxis_title='z (m)',aspectmode='data'))
    return fig


def show_field(result, choices=None, default_field=None, key_prefix='fields'):
    names = [k for k in result['fields'] if choices is None or choices(k)]
    if not names:return
    default_index = names.index(default_field) if default_field in names else 0
    chosen=st.selectbox('Grandeur représentée',names,index=default_index,key=f'{key_prefix}_select')
    field=result['fields'][chosen]
    if result['dimension']==2:
        st.plotly_chart(chart_2d(field,result),use_container_width=True)
    else:
        try:
            plot_3d_pyvista(result,chosen)
            st.caption('Vue PyVista interactive : rotation, zoom, trois coupes orthogonales du volume.')
        except Exception as exc:
            st.info(f'Rendu PyVista indisponible sur ce serveur : visualisation Plotly de secours. ({type(exc).__name__})')
            st.plotly_chart(plot_3d_fallback(result,chosen),use_container_width=True)
    a=field['data']; x,y=st.columns(2)
    x.metric(f'Minimum ({field["unit"]})',f'{np.min(a):.4g}')
    y.metric(f'Maximum ({field["unit"]})',f'{np.max(a):.4g}')


def csv_of(columns):
    return pd.DataFrame(columns).to_csv(index=False).encode('utf-8-sig')


def packaged_export(result):
    sink=io.BytesIO()
    with zipfile.ZipFile(sink,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('parametres_SI.json',json.dumps(result['params'],indent=2,ensure_ascii=False))
        z.writestr('historique_thermique.csv',csv_of({'temps_h':result['times_h'],
             'Tmax_C':result['tmax_C'],'Tcentre_C':result['tcentre_C'],
             'E_GPa':result['young_GPa'],'fcm_MPa':result['resistance_MPa']}))
        z.writestr('historique_potentiel_RSI.csv',csv_of({'temps_h':result['times_epsilon_h'],'epsilon_max_pct':result['epsilon_max_percent']}))
        z.writestr('champs_arrays.npz',npz_bytes(result))
        z.writestr('README.txt','Champs stockés en ordre (y,x) pour 2D et (z,y,x) pour 3D.\n'+
            'Champs aux cellules ou aux noeuds : voir champs_metadata.json.\n'+
            'Contraintes : convention de signe du script d’origine conservée.\n')
        z.writestr('champs_metadata.json',json.dumps({name:{'array_key':f'f{i:02d}', 'unit':f['unit'],'position':f['position'],'shape':list(f['data'].shape)} for i,(name,f) in enumerate(result['fields'].items())},ensure_ascii=False,indent=2))
        if result['dimension']==3:
            try:
                grid,_=create_pyvista_grid(result,'Température au pic thermique')
                # Stocker également tous les champs en tant que jeux de données VTK.
                for name,f in result['fields'].items():
                    entry=f'{name} [{f["unit"]}]'
                    if f['position']=='point':grid.point_data[entry]=f['data'].ravel(order='C')
                    else:grid.cell_data[entry]=f['data'].ravel(order='C')
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'simulation_3d.vti'
                    grid.save(p)
                    z.write(p,arcname='simulation_3d.vti')
            except (ImportError,RuntimeError,OSError,ValueError):
                pass
    return sink.getvalue()


def npz_bytes(result):
    sink=io.BytesIO()
    np.savez_compressed(sink,**{f'f{i:02d}':v['data'] for i,v in enumerate(result['fields'].values())})
    return sink.getvalue()


st.markdown('<div class="page-kicker">Béton massif · Thermochimie et RSI</div>',unsafe_allow_html=True)
st.title('Simulation thermique et potentiel de RSI')
st.markdown('<p class="hero-subtitle">Interface Streamlit · Résultats Plotly et PyVista · Moteur numérique corrigé (étape 1)</p>',unsafe_allow_html=True)
submitted, params=parameters_form()

if submitted:
    current=st.progress(0,text='Calcul en cours…')
    try:
        def progress(x):current.progress(max(0,min(1,float(x))),text=f'Calcul en cours… {100*x:.0f} %')
        result=run_simulation(params,progress_callback=progress)
        st.session_state['rsi_result']=result
        current.empty()
        st.success(f'Calcul terminé en {result["run_time_s"]:.2f} s. Résultats disponibles ci-dessous.')
    except (ValueError,RuntimeError,MemoryError,FloatingPointError) as exc:
        current.empty()
        st.error(f'Le calcul n’a pas abouti : {exc}')

result=st.session_state.get('rsi_result')
if result is None:
    st.info('Choisis la dimension, la géométrie et les paramètres dans la barre latérale, puis lance une simulation. Les résultats resteront accessibles durant cette session.')
    a,b,c=st.columns(3)
    a.markdown('**01 — Paramètres**\n\nGéométrie et propriétés du béton.')
    b.markdown('**02 — Simulation**\n\nCalcul thermique, mécanique et RSI.')
    c.markdown('**03 — Exploitation**\n\nCartes, courbes, coupe 3D et export.')
    st.stop()

n=result['dimension']; L,W,*other=result['dimensions_m']
st.caption(f'Dernier calcul : {n}D · {L:g} × {W:g}'+(f' × {other[0]:g}' if n==3 else '')+f' m · {result["run_time_s"]:.1f} s · durée simulée : {result["times_h"][-1]:.1f} h')
col1,col2,col3,col4=st.columns(4)
col1.metric('T max (°C)',f'{result["peak_temperature_C"]:.1f}')
col2.metric('Pic atteint à (h)',f'{result["peak_temperature_time_h"]:.1f}')
col3.metric('ε∞ RSI maximale (%)',f'{result["max_epsilon_percent"]:.4f}')
col4.metric('Pas de temps (s)',f'{result["dt_s"]:.1f}')

summary,thermal,rsi,mechanics,exports=st.tabs(['Vue générale','Thermique','Potentiel RSI','Mécanique','Exporter'])
with summary:
    left,right=st.columns([1.2,1])
    with left:
        st.subheader('Évolution des températures')
        st.plotly_chart(graph_history(result,'temperature'),use_container_width=True)
    with right:
        st.subheader('Évolution du potentiel RSI')
        st.plotly_chart(graph_history(result,'epsilon'),use_container_width=True)
    st.caption('Le champ de RSI représente un potentiel de gonflement ultime estimé à partir de l’histoire thermique jeune âge. Ce n’est pas une prévision temporelle du gonflement réel.')
with thermal:
    st.subheader('Cartographie thermique')
    show_field(result,choices=lambda name: name.startswith(('Température','Degré')),default_field='Température au pic thermique',key_prefix='thermal')
    st.subheader('Évolution des propriétés du béton')
    st.plotly_chart(graph_history(result,'beton'),use_container_width=True)
with rsi:
    st.subheader('Potentiel de déformation expansive ultime')
    show_field(result,choices=lambda name: name=='Déformation RSI ultime' or name.startswith(('Déplacement RSI','Contrainte RSI')),
               default_field='Déformation RSI ultime',key_prefix='rsi')
    st.caption('Les contraintes RSI sont calculées dans le modèle mécanique élastique conservé, sans cinétique pluriannuelle de gonflement, humidité ni fissuration.')
with mechanics:
    st.subheader('Déplacements et contraintes du béton au jeune âge')
    show_field(result,choices=lambda name: '(Jeune âge' in name, key_prefix='mech')
    st.caption('Les cartes jeune âge affichent les champs de l’instant où le maximum absolu de chaque composante a été enregistré ; les cartes peuvent donc correspondre à des instants distincts.')
with exports:
    st.subheader('Télécharger les résultats')
    st.write('L’archive comprend paramètres, historiques, champs et métadonnées. En 3D, le fichier VTI est également inclus lorsque PyVista est disponible sur le serveur.')
    st.download_button('Télécharger l’étude complète (.zip)',packaged_export(result),file_name='rsi_resultats.zip',mime='application/zip',type='primary')
    st.download_button('Télécharger l’historique thermique (.csv)',csv_of({'temps_h':result['times_h'],'Tmax_C':result['tmax_C'],'Tcentre_C':result['tcentre_C']}),
                       file_name='historique_thermique.csv',mime='text/csv')
    st.json(result['params'],expanded=False)

st.divider()
st.caption('Prototype de visualisation — Ce logiciel ne constitue pas, à ce stade, une validation d’ingénierie de l’ouvrage ou du risque effectif de RSI.')
