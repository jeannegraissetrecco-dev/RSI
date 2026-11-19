"""Moteur de calcul issu de FINAL_corrige_etape1.py (étape 1).

Seule la couche Tkinter et les figures Matplotlib ont été retirées.
Les modèles et algorithmes restent ceux de la version corrigée.
Le calcul est exécuté dans une fonction, sans état global partagé.
"""
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def _check_and_prepare(p):
    values = dict(p)
    dim = int(values.get('dim', 2))
    if dim not in (2, 3):
        raise ValueError('Dimension : sélectionner 2D ou 3D.')
    numeric_positive = ['lam', 'C_p', 'rho', 'Efin', 'longueur', 'largeur', 'hx', 'hy', 'duree']
    if dim == 3:
        numeric_positive += ['hauteur', 'hz']
    for key in numeric_positive:
        x = float(values[key])
        if not np.isfinite(x) or x <= 0:
            raise ValueError(f'{key} doit être un nombre strictement positif.')
    if not 0 <= float(values['nu']) < 0.5:
        raise ValueError('Le coefficient de Poisson doit être entre 0 et 0,5 exclu.')
    for length, step in [('longueur', 'hx'), ('largeur','hy')] + ([('hauteur','hz')] if dim == 3 else []):
        n = float(values[length])/float(values[step])
        if n < 3:
            raise ValueError(f'Prévoir au moins 3 cellules selon {length} pour les cartes.')
        if not np.isclose(n, round(n), rtol=0, atol=1e-7):
            raise ValueError(f'{length} doit être un multiple entier de {step} dans cette version.')
    cells = int(round(values['longueur']/values['hx'])) * int(round(values['largeur']/values['hy']))
    if dim == 3:
        cells *= int(round(values['hauteur']/values['hz']))
    ceiling = 1600 if dim==2 else 216
    if cells > ceiling:
        raise ValueError(f'Maillage trop dense pour le prototype Web : {cells} cellules (limite : {ceiling}). Affiner localement à l’étape optimisation.')
    if float(values['duree']) <= 0 or not np.isfinite(float(values['duree'])):
        raise ValueError('La durée doit être positive.')
    if float(values['drc']) < 0:
        raise ValueError('La date de décoffrage doit être positive.')
    return values


def run_simulation(params, progress_callback=None):
    """Exécuter le modèle historique corrigé et renvoyer des données exploitable par Plotly/PyVista.

    params en unités SI : dimensions en m, temps en s, températures en K, Efin en Pa.
    Le serveur ne doit PAS partager les variables de calcul entre utilisateurs.
    """
    p = _check_and_prepare(params)
    dim = int(p['dim'])
    lam = float(p['lam'])
    C_p = float(p['C_p'])
    rho = float(p['rho'])
    nu = float(p['nu'])
    alpha = float(p['alpha'])
    Efin = float(p['Efin'])
    T0 = float(p['T0'])
    Text = float(p['Text'])
    duree = float(p['duree'])
    deltatmin = float(p['deltatmin'])
    longueur = float(p['longueur'])
    largeur = float(p['largeur'])
    hauteur = float(p.get('hauteur', 0.0))
    drc = float(p['drc'])
    hair = float(p['hair'])
    hx = float(p['hx']); hy = float(p['hy']); hz = float(p.get('hz', 1.0))
    clxinf = p['clxinf']; clxsup = p['clxsup']
    clyinf = p['clyinf']; clysup = p['clysup']
    clzinf = p.get('clzinf','Symétrie'); clzsup = p.get('clzsup','Symétrie')
    clxinf2 = p['clxinf2']; clxsup2 = p['clxsup2']
    clyinf2 = p['clyinf2']; clysup2 = p['clysup2']
    clzinf2 = p.get('clzinf2','Symétrie'); clzsup2 = p.get('clzsup2','Symétrie')
    D = lam / (C_p * rho)

    tps = []
    temps = 0
    realtime = time.time()
    Qinf = 71250
    Ea_sur_R = 4537
    T_seuil = 54 + 273.15
    #epsilon_m_point = 0.00133 # par heure
    epsilon_m_point = 0.00133 / 3600 # par seconde
    E_a_Diss = 407.8 # J par mol
    R = 8.314

    def E(x):
        return Efin * x

    def mu(x) :
        return E(x) / (2*(1+nu))

    def lmbda(x) :
        return E(x) * nu /((1+nu)*(1-2*nu))

    def get_h_value(cl):
        if cl == 'Coffrage bois':
            return 5
        elif cl == 'Coffrage tôle' or cl == 'Coffrage métallique' or cl == 'Bord libre':
            return hair
        elif cl == 'Pièce béton':
            return 1
        elif cl == 'Symétrie' or cl == 'symétrie':
            return 0
        else:
            return 0  # Valeur par défaut si la condition n'est pas reconnue

    def resbet(t): # t est en heures
        tj = t / 24
        return (np.exp(0.25*(1-(28/tj)**(1/2))))*43


    # ========== CORRECTIONS NUMERIQUES COMMUNES 2D/3D ==========
    def h_effectif(h, pas):
        """Coefficient d'echange vu depuis le centre d'une cellule de bord (W/m²/K)."""
        if h <= 0:
            return 0.0
        return 1.0 / (1.0 / h + pas / (2.0 * lam))


    def pas_thermique_2d(T_prev, xi_prev, pas_x, pas_y, conditions, dt):
        """Bilan explicite volumes finis, y compris aux bords et dans les coins.

        T_prev et xi_prev sont des champs *aux centres des cellules* (y, x).
        conditions = (h_gauche, h_droite, h_bas, h_haut).
        """
        h_g, h_d, h_b, h_h = conditions
        variation = np.zeros_like(T_prev)
        cx = D * dt / pas_x**2
        cy = D * dt / pas_y**2
        # Flux intercellulaires conservatifs ; axes de T[y, x].
        variation[:, 1:] += cx * (T_prev[:, :-1] - T_prev[:, 1:])
        variation[:, :-1] += cx * (T_prev[:, 1:] - T_prev[:, :-1])
        variation[1:, :] += cy * (T_prev[:-1, :] - T_prev[1:, :])
        variation[:-1, :] += cy * (T_prev[1:, :] - T_prev[:-1, :])
        conv_x = dt / (rho * C_p * pas_x)
        conv_y = dt / (rho * C_p * pas_y)
        variation[:, 0] += conv_x * h_effectif(h_g, pas_x) * (Text - T_prev[:, 0])
        variation[:, -1] += conv_x * h_effectif(h_d, pas_x) * (Text - T_prev[:, -1])
        variation[0, :] += conv_y * h_effectif(h_b, pas_y) * (Text - T_prev[0, :])
        variation[-1, :] += conv_y * h_effectif(h_h, pas_y) * (Text - T_prev[-1, :])
        xi_dot = AffiniteChimique(xi_prev) * np.exp(-Ea_sur_R / T_prev)
        # Limiter la chaleur liberee au meme avancement que l'hydratation.
        # Sinon, un pas trop grand peut creer plus que Qinf de chaleur.
        delta_xi = np.clip(dt * xi_dot, 0.0, np.maximum(1.0 - xi_prev, 0.0))
        T_next = T_prev + variation + Qinf / C_p * delta_xi
        xi_next = xi_prev + delta_xi
        return T_next, xi_next


    def pas_thermique_3d(T_prev, xi_prev, pas_x, pas_y, pas_z, conditions, dt):
        """Bilan conservatif pour T[z,y,x] ; chaque face reçoit sa source d'hydratation."""
        h_g, h_d, h_av, h_ar, h_b, h_h = conditions
        variation = np.zeros_like(T_prev)
        cx = D * dt / pas_x**2
        cy = D * dt / pas_y**2
        cz = D * dt / pas_z**2
        variation[:, :, 1:] += cx * (T_prev[:, :, :-1] - T_prev[:, :, 1:])
        variation[:, :, :-1] += cx * (T_prev[:, :, 1:] - T_prev[:, :, :-1])
        variation[:, 1:, :] += cy * (T_prev[:, :-1, :] - T_prev[:, 1:, :])
        variation[:, :-1, :] += cy * (T_prev[:, 1:, :] - T_prev[:, :-1, :])
        variation[1:, :, :] += cz * (T_prev[:-1, :, :] - T_prev[1:, :, :])
        variation[:-1, :, :] += cz * (T_prev[1:, :, :] - T_prev[:-1, :, :])
        conv_x = dt / (rho * C_p * pas_x)
        conv_y = dt / (rho * C_p * pas_y)
        conv_z = dt / (rho * C_p * pas_z)
        variation[:, :, 0] += conv_x * h_effectif(h_g, pas_x) * (Text - T_prev[:, :, 0])
        variation[:, :, -1] += conv_x * h_effectif(h_d, pas_x) * (Text - T_prev[:, :, -1])
        variation[:, 0, :] += conv_y * h_effectif(h_av, pas_y) * (Text - T_prev[:, 0, :])
        variation[:, -1, :] += conv_y * h_effectif(h_ar, pas_y) * (Text - T_prev[:, -1, :])
        variation[0, :, :] += conv_z * h_effectif(h_b, pas_z) * (Text - T_prev[0, :, :])
        variation[-1, :, :] += conv_z * h_effectif(h_h, pas_z) * (Text - T_prev[-1, :, :])
        xi_dot = AffiniteChimique(xi_prev) * np.exp(-Ea_sur_R / T_prev)
        # Limiter la chaleur liberee au meme avancement que l'hydratation.
        # Sinon, un pas trop grand peut creer plus que Qinf de chaleur.
        delta_xi = np.clip(dt * xi_dot, 0.0, np.maximum(1.0 - xi_prev, 0.0))
        T_next = T_prev + variation + Qinf / C_p * delta_xi
        xi_next = xi_prev + delta_xi
        return T_next, xi_next


    def base_mouvements_rigides(coords):
        """Modes de translation et rotation rigides, lignes ordonnees par ddl global."""
        nd = coords.shape[1]
        modes = np.zeros((nd * len(coords), 3 if nd == 2 else 6))
        if nd == 2:
            x, y = coords.T
            modes[0::2, 0] = 1
            modes[1::2, 1] = 1
            modes[0::2, 2] = -y
            modes[1::2, 2] = x
        else:
            x, y, z = coords.T
            modes[0::3, 0] = modes[1::3, 1] = modes[2::3, 2] = 1
            modes[1::3, 3], modes[2::3, 3] = -z, y  # Rx
            modes[0::3, 4], modes[2::3, 4] = z, -x  # Ry
            modes[0::3, 5], modes[1::3, 5] = -y, x  # Rz
        return modes


    def completer_blocages_rigides(bc_dofs, coords, modes):
        """Ajoute juste les ddl necessaires pour supprimer les modes rigides restants.

        Les blocages deja prescrits sont conserves. Les ddl ajoutes representent uniquement
        un choix de repere : chaque ajout augmente le rang des modes rigides elimines.
        """
        bc = set(bc_dofs)
        rang = np.linalg.matrix_rank(modes[sorted(bc), :]) if bc else 0
        if rang == modes.shape[1]:
            return sorted(bc)
        nd = coords.shape[1]
        # Choisir quelques coins distants d'abord, pour un ensemble de blocages stable.
        candidates_nodes = [0, len(coords)-1] + [int(np.argmax(coords[:, k])) for k in range(nd)]
        candidates = [nd * n + k for n in candidates_nodes for k in range(nd)]
        candidates.extend(range(nd * len(coords)))
        for dof_id in candidates:
            if dof_id in bc:
                continue
            nouveau_rang = np.linalg.matrix_rank(modes[sorted(bc | {dof_id}), :])
            if nouveau_rang > rang:
                bc.add(dof_id)
                rang = nouveau_rang
            if rang == modes.shape[1]:
                break
        if rang < modes.shape[1]:
            raise ValueError('Impossible de supprimer les mouvements rigides : maillage degenere.')
        return sorted(bc)


    def appliquer_blocages_nuls(K, F, bc_dofs):
        """Dirichlet homogene : annule lignes ET colonnes (matrice symetrique)."""
        for dof_id in sorted(set(bc_dofs)):
            K[dof_id, :] = 0
            K[:, dof_id] = 0
            K[dof_id, dof_id] = 1
            F[dof_id] = 0


    def C_2d_depuis_xi(xi, nu):
        coeff = E(xi) / ((1 + nu) * (1 - 2 * nu))
        return coeff[..., None, None] * np.array([
            [1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu)/2]
        ])


    def C_3d_depuis_xi(xi):
        lb, m = lmbda(xi), mu(xi)
        C = np.zeros(xi.shape + (6, 6))
        for a in range(3):
            C[..., a, a] = lb + 2 * m
            for b in range(3):
                if b != a:
                    C[..., a, b] = lb
        for a in range(3, 6):
            C[..., a, a] = m
        return C


    def transferer_C_2d(C, C_node, C_weights):
        C_node.fill(0)
        C_weights.fill(0)
        ny, nx = C.shape[:2]
        for dj in (0, 1):
            for di in (0, 1):
                C_node[dj:ny+dj, di:nx+di] += C
                C_weights[dj:ny+dj, di:nx+di] += 1
        C_node /= C_weights[..., None, None]


    def transferer_C_3d(C, C_node, C_weights):
        C_node.fill(0)
        C_weights.fill(0)
        nz, ny, nx = C.shape[:3]
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    C_node[dk:nz+dk, dj:ny+dj, di:nx+di] += C
                    C_weights[dk:nz+dk, dj:ny+dj, di:nx+di] += 1
        C_node /= C_weights[..., None, None]

    if dim == 2:
        # hx = 0.05
        # hy = 0.05
        Ny = int(largeur/hy)
        Nx = int(longueur/hx)
        nnx = Nx + 1
        nny = Ny + 1
        nn = nnx * nny
        ne = Nx * Ny
        coords = np.array([[i * hx, j * hy] for j in range(nny) for i in range(nnx)])
        dt = 0.1* min(hx**2, hy**2) / D
        t = np.ones((Ny, Nx)) * T0
        xi = np.zeros((Ny, Nx))
        adexi = np.zeros((Ny, Nx))
        hxinf = get_h_value(clxinf)
        hxsup = get_h_value(clxsup)
        hyinf = get_h_value(clyinf)
        hysup = get_h_value(clysup)
        hxinf2 = get_h_value(clxinf2)
        hxsup2 = get_h_value(clxsup2)
        hyinf2 = get_h_value(clyinf2)
        hysup2 = get_h_value(clysup2)
        liste_temps = []
        liste_temps_meca = []
        liste_E = []
        liste_resbet = []
        liste_tmax = []
        liste_tcentre = []
        liste_uxmax_pre = []
        liste_uxmin_pre = []
        liste_uymin_pre = []
        liste_uymax_pre = []
        liste_sxmin_pre = []
        liste_symin_pre = []
        liste_sxmax_pre = []
        liste_symax_pre = []
        liste_uxmax_post = []
        liste_uxmin_post = []
        liste_uymin_post = []
        liste_uymax_post = []
        liste_sxmin_post = []
        liste_symin_post = []
        liste_sxmax_post = []
        liste_symax_post = []
        temperature_max = 0
        temps_temperature_max = 0
        abs_ux_max_pre = 0
        temps_abs_ux_max_pre = 0
        abs_uy_max_pre = 0
        temps_abs_uy_max_pre = 0
        abs_sx_max_pre = 0
        temps_abs_sx_max_pre = 0
        abs_sy_max_pre = 0
        temps_abs_sy_max_pre = 0
        abs_ux_max_post = 0
        temps_abs_ux_max_post = 0
        abs_uy_max_post = 0
        temps_abs_uy_max_post = 0
        abs_sx_max_post = 0
        temps_abs_sx_max_post = 0
        abs_sy_max_post = 0
        temps_abs_sy_max_post = 0
        grille_tmax = np.zeros((Ny, Nx))
        grille_uxmax_pre = np.zeros((nny,nnx))
        grille_uymax_pre = np.zeros((nny,nnx))
        grille_sxmax_pre = np.zeros((nny,nnx))
        grille_symax_pre = np.zeros((nny,nnx))
        grille_uxmax_post = np.zeros((nny,nnx))
        grille_uymax_post = np.zeros((nny,nnx))
        grille_sxmax_post = np.zeros((nny,nnx))
        grille_symax_post = np.zeros((nny,nnx))
        c = np.zeros((Ny, Nx))
        C = np.zeros((Ny, Nx, 3, 3))
        epsilon_inf = np.zeros((nny, nnx))
        u_x = np.zeros((nny, nnx))
        u_y = np.zeros((nny, nnx))
        s_xx = np.zeros((Ny, Nx))
        s_yy = np.zeros((Ny, Nx))
        T_node = np.zeros((nny, nnx))
        C_node = np.zeros((nny, nnx, 3, 3))
        T_weights = np.zeros((nny, nnx))
        C_weights = np.zeros((nny, nnx))
        liste_epsilon_max = []
        liste_temps_epsilon_max = []

        def node_id(i, j):
            return j * nnx + i

        def AffiniteChimique(ksi):
            hydrabis = np.array([0, 0.1091, 0.1259, 0.1259, 0.1301, 0.1371, 0.1483, 0.1538, 0.1636, 0.1734, 0.1832, 0.1930, 0.2028, 0.2126, 0.2224, 0.2322, 0.2420, 0.2518, 0.2616, 0.2714, 0.2812, 0.2910, 0.3008, 0.3106, 0.3203, 0.3301, 0.3399, 0.3497, 0.3595, 0.3693, 0.3791, 0.3889, 0.3987, 0.4085, 0.4183, 0.4281, 0.4477, 0.4673, 0.4869, 0.5065, 0.5261, 0.6855, 0.8183, 0.9752, 1])
            affchim = np.array([51.47, 51.47, 380.88, 439.22, 490.69, 545.59, 603.92, 665.69, 782.35, 840.69, 888.73, 933.33, 977.94, 1011.27, 1037.25, 1059.80, 1071.57, 1069.61, 1063.73, 1049.02, 1033.33, 1007.84, 978.43, 945.10, 909.80, 870.10, 832.35, 789.22, 746.08, 697.55, 650.98, 605.39, 560.29, 516.67, 473.04, 439.22, 368.14, 314.22, 265.20, 228.43, 187.75, 124.02, 0, 0, 0])
            result = np.zeros_like(ksi)
            mask = ksi < 1
            result[mask] = np.interp(ksi[mask], hydrabis, affchim) * 0.171
            return result

        elements = []
        for j in range(Ny):
            for i in range(Nx):
                n1 = node_id(i, j)
                n2 = node_id(i+1, j)
                n3 = node_id(i+1, j+1)
                n4 = node_id(i, j+1)
                elements.append([n1, n2, n3, n4])

        dof = 2 * nn
        start_time = time.time()
        iteration = 0

        # Pré-calculer certaines valeurs constantes
        precomputed_B_template = np.zeros((3, 8))
        precomputed_J = np.array([[hx/2, 0], [0, hy/2]])
        detJ = np.linalg.det(precomputed_J)
        invJ = np.linalg.inv(precomputed_J)
        dN_dxi = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]) / 4
        dN_dx = invJ @ dN_dxi
        coeff_diff_x = lam * dt / (rho * C_p * hx * hx)
        coeff_diff_y = lam * dt / (rho * C_p * hy * hy)

        modes_rigides = base_mouvements_rigides(coords)
        while temps < duree and (np.max(t) - 273.15 > deltatmin or temps < 172800*2):
            # La loi thermique applique vraiment les conditions avant/apres decoffrage.
            etat_apres_decoffrage = (temps >= drc)
            cl_thermiques = (hxinf2, hxsup2, hyinf2, hysup2) if etat_apres_decoffrage else (hxinf, hxsup, hyinf, hysup)
            t, xi = pas_thermique_2d(t, xi, hx, hy, cl_thermiques, dt)

            Tmax = np.max(t) - 273.15
            T_centre = t[Ny//2, Nx//2] - 273.15
            if np.max(t) > temperature_max :
                grille_tmax = t.copy()
                temperature_max = np.max(t)
                temps_temperature_max = temps

            liste_temps.append(temps / 3600)
            liste_E.append(E(np.max(xi)) * 1e-9)
            liste_resbet.append(resbet((temps+1)/3600))
            liste_tmax.append(Tmax)
            liste_tcentre.append(T_centre)

            # Passage cellule -> noeud pour la temperature a chaque pas thermique.
            T_node.fill(0)
            T_weights.fill(0)
            for dj in (0, 1):
                for di in (0, 1):
                    T_node[dj:Ny+dj, di:Nx+di] += t
                    T_weights[dj:Ny+dj, di:Nx+di] += 1
            T_node /= T_weights

            # Gonflement potentiel RSI : actualisation et historique une fois par pas.
            masque_chaud = T_node > T_seuil
            epsilon_inf[masque_chaud] += epsilon_m_point * dt * np.exp(
                -E_a_Diss / (R * (T_node[masque_chaud] - T_seuil))
            )
            liste_epsilon_max.append(float(np.max(epsilon_inf)))
            liste_temps_epsilon_max.append(temps / 3600)

            if iteration % 100 == 0:
                liste_temps_meca.append(temps / 3600)
                # IMPORTANT : d'abord calculer C au temps present, ensuite interpoler C_node.
                C = C_2d_depuis_xi(xi, nu)
                transferer_C_2d(C, C_node, C_weights)

                # Éléments finis, partie 1
                K = lil_matrix((dof, dof))
                F = np.zeros(dof)
                B_template = np.zeros((3, 8))
                strain_elastic = []
                stress = []
                centers = []

                # # --- Assemblage éléments finis avec quadrature de Gauss 2x2 ---
                # K = lil_matrix((dof, dof))
                # F = np.zeros(dof)
                # strain_elastic = []
                # stress = []
                # centers = []

                # Paramètres de quadrature 2x2
                gp = 1.0 / np.sqrt(3.0)
                gauss_pts = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]
                weights = [1.0, 1.0, 1.0, 1.0]

                # Fonctions de forme et dérivées
                def shape_dN_dxi(xa, eta):
                    N = 0.25 * np.array([
                        (1 - xa) * (1 - eta),
                        (1 + xa) * (1 - eta),
                        (1 + xa) * (1 + eta),
                        (1 - xa) * (1 + eta)
                    ])
                    dN_dxi = 0.25 * np.array([
                        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],  # dN/dxi
                        [-(1 - xa), -(1 + xa), (1 + xa),  (1 - xa)]       # dN/deta
                    ])
                    return N, dN_dxi

                # Boucle sur les éléments
                for e, conn in enumerate(elements):
                    xe = coords[conn]
                    #print(xe)  # coordonnées (4x2)
                    Te = T_node[np.unravel_index(conn, (nny, nnx))]
                    x_c = np.mean(xe, axis=0)
                    dT = np.mean(Te) - T0
                    C_Utile = np.mean(C_node[np.unravel_index(conn, (nny, nnx))], axis=0)
                    centers.append(x_c)
                    Ke = np.zeros((8, 8))
                    fe_th = np.zeros(8)

                    # Intégration 2x2 de Gauss
                    for (xa, eta), w in zip(gauss_pts, weights):
                        N, dN_dxi = shape_dN_dxi(xa, eta)
                        J = dN_dxi @ xe
                        detJ = np.linalg.det(J)
                        invJ = np.linalg.inv(J)
                        dN_dx = invJ @ dN_dxi

                        # Matrice B (déformations planes)
                        B = np.zeros((3, 8))
                        for i in range(4):
                            B[0, 2*i  ] = dN_dx[0, i]
                            B[1, 2*i+1] = dN_dx[1, i]
                            B[2, 2*i  ] = dN_dx[1, i]
                            B[2, 2*i+1] = dN_dx[0, i]

                        # Contribution de rigidité
                        Ke += (B.T @ C_Utile @ B) * detJ * w
                        #print(w)

                        # Effet thermique
                        eps_th = (1 + nu) * alpha * dT * np.array([1, 1, 0])  # deformations planes, effet epsilon_zz thermique
                        fe_th += (B.T @ C_Utile @ eps_th) * detJ * w

                    # Indices de degrés de liberté globaux
                    dof_indices = []
                    for n in conn:
                        dof_indices.extend([2 * n, 2 * n + 1])

                    # Assemblage global
                    for i in range(8):
                        F[dof_indices[i]] += fe_th[i]
                        for j in range(8):
                            K[dof_indices[i], dof_indices[j]] += Ke[i, j]

                # Conditions aux limites MECANIQUES : independantes des coefficients h.
                # Avant decoffrage : coffrage bloque ici tous les ddl de la face (hypothese
                # initiale conservee). Apres decoffrage : utiliser les choix definitifs.
                conds = ((clxinf2, clxsup2, clyinf2, clysup2) if temps >= drc
                         else (clxinf, clxsup, clyinf, clysup))
                bc_dofs = []
                faces = (
                    (conds[0], [node_id(0, j) for j in range(nny)], 0),
                    (conds[1], [node_id(nnx-1, j) for j in range(nny)], 0),
                    (conds[2], [node_id(i, 0) for i in range(nnx)], 1),
                    (conds[3], [node_id(i, nny-1) for i in range(nnx)], 1),
                )
                for condition, nodes, normale in faces:
                    if condition.lower() == 'symétrie':
                        bc_dofs.extend(2*n + normale for n in nodes)
                    elif condition != 'Bord libre':
                        bc_dofs.extend(2*n + k for n in nodes for k in range(2))
                bc_dofs = completer_blocages_rigides(bc_dofs, coords, modes_rigides)
                appliquer_blocages_nuls(K, F, bc_dofs)

                K = csr_matrix(K)
                u = spsolve(K, F)

                for e, conn in enumerate(elements):
                    Te = T_node[np.unravel_index(conn, (nny, nnx))]
                    xe = coords[conn]
                    dT = np.mean(Te) - T0
                    C_Utile = np.mean(C_node[np.unravel_index(conn, (nny, nnx))], axis=0)
                    ue = np.zeros(8)
                    for i in range(4):
                        ue[2*i] = u[2*conn[i]]
                        ue[2*i+1] = u[2*conn[i]+1]
                    dN_dxi = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]) / 4
                    J = dN_dxi @ xe
                    invJ = np.linalg.inv(J)
                    dN_dx = invJ @ dN_dxi
                    B = B_template.copy()
                    for i in range(4):
                        B[0, 2*i  ] = dN_dx[0, i]
                        B[1, 2*i+1] = dN_dx[1, i]
                        B[2, 2*i  ] = dN_dx[1, i]
                        B[2, 2*i+1] = dN_dx[0, i]
                    eps_total = B @ ue
                    eps_th = (1 + nu) * alpha * dT * np.array([1, 1, 0])  # contrainte therm. hors-plan incluse
                    eps_el = eps_total - eps_th
                    sigma = C_Utile @ eps_el
                    # Deformation ELASTIQUE physique dans le plan (differente de eps_eff).
                    strain_elastic.append(eps_total - alpha * dT * np.array([1, 1, 0]))
                    stress.append(sigma)

                u_x = u[0::2].reshape((nny, nnx))
                u_y = u[1::2].reshape((nny, nnx))
                s_xx = np.array([-s[0] for s in stress]).reshape((Ny, Nx))
                s_yy = np.array([-s[1] for s in stress]).reshape((Ny, Nx))

                liste_uxmax_pre.append(np.max(u_x[1:-1,1:-1]))
                liste_uxmin_pre.append(np.min(u_x[1:-1,1:-1]))
                liste_uymin_pre.append(np.min(u_y[1:-1,1:-1]))
                liste_uymax_pre.append(np.max(u_y[1:-1,1:-1]))
                liste_sxmin_pre.append(np.min(s_xx[1:-1,1:-1]))
                liste_symin_pre.append(np.min(s_yy[1:-1,1:-1]))
                liste_sxmax_pre.append(np.max(s_xx[1:-1,1:-1]))
                liste_symax_pre.append(np.max(s_yy[1:-1,1:-1]))
                if temps > drc :
                    liste_uxmax_post.append(np.max(u_x[1:-1,1:-1]))
                    liste_uxmin_post.append(np.min(u_x[1:-1,1:-1]))
                    liste_uymin_post.append(np.min(u_y[1:-1,1:-1]))
                    liste_uymax_post.append(np.max(u_y[1:-1,1:-1]))
                    liste_sxmin_post.append(np.min(s_xx[1:-1,1:-1]))
                    liste_symin_post.append(np.min(s_yy[1:-1,1:-1]))
                    liste_sxmax_post.append(np.max(s_xx[1:-1,1:-1]))
                    liste_symax_post.append(np.max(s_yy[1:-1,1:-1]))
                else :
                    liste_uxmax_post.append(0)
                    liste_uxmin_post.append(0)
                    liste_uymin_post.append(0)
                    liste_uymax_post.append(0)
                    liste_sxmin_post.append(0)
                    liste_symin_post.append(0)
                    liste_sxmax_post.append(0)
                    liste_symax_post.append(0)

                if np.max(np.abs(u_x[1:-1,1:-1])) > abs_ux_max_pre:
                    grille_uxmax_pre = u_x.copy()
                    abs_ux_max_pre = np.max(np.abs(u_x[1:-1,1:-1]))
                    temps_abs_ux_max_pre = temps
                if np.max(np.abs(u_y[1:-1,1:-1])) > abs_uy_max_pre:
                    grille_uymax_pre = u_y.copy()
                    abs_uy_max_pre = np.max(np.abs(u_y[1:-1,1:-1]))
                    temps_abs_uy_max_pre = temps
                if np.max(np.abs(s_xx[1:-1,1:-1])) > abs_sx_max_pre:
                    grille_sxmax_pre = s_xx.copy()
                    abs_sx_max_pre = np.max(np.abs(s_xx[1:-1,1:-1]))
                    temps_abs_sx_max_pre = temps
                if np.max(np.abs(s_yy[1:-1,1:-1])) > abs_sy_max_pre:
                    grille_symax_pre = s_yy.copy()
                    abs_sy_max_pre = np.max(np.abs(s_yy[1:-1,1:-1]))
                    temps_abs_sy_max_pre = temps

                if temps > drc :
                    if np.max(np.abs(u_x[1:-1,1:-1])) > abs_ux_max_post:
                        grille_uxmax_post = u_x.copy()
                        abs_ux_max_post = np.max(np.abs(u_x[1:-1,1:-1]))
                        temps_abs_ux_max_post = temps
                    if np.max(np.abs(u_y[1:-1,1:-1])) > abs_uy_max_post:
                        grille_uymax_post = u_y.copy()
                        abs_uy_max_post = np.max(np.abs(u_y[1:-1,1:-1]))
                        temps_abs_uy_max_post = temps
                    if np.max(np.abs(s_xx[1:-1,1:-1])) > abs_sx_max_post:
                        grille_sxmax_post = s_xx.copy()
                        abs_sx_max_post = np.max(np.abs(s_xx[1:-1,1:-1]))
                        temps_abs_sx_max_post = temps
                    if np.max(np.abs(s_yy[1:-1,1:-1])) > abs_sy_max_post:
                        grille_symax_post = s_yy.copy()
                        abs_sy_max_post = np.max(np.abs(s_yy[1:-1,1:-1]))
                        temps_abs_sy_max_post = temps

            iteration += 1
            temps += dt


        vminupre = min(np.min(liste_uxmin_pre), np.min(liste_uymin_pre))
        vmaxupre = max(np.max(liste_uxmax_pre), np.max(liste_uymax_pre))
        vminspre = min(np.min(liste_sxmin_pre), np.min(liste_symin_pre))
        vmaxspre = max(np.max(liste_sxmax_pre), np.max(liste_symax_pre))
        vminupost = min(np.min(liste_uxmin_post), np.min(liste_uymin_post))
        vmaxupost = max(np.max(liste_uxmax_post), np.max(liste_uymax_post))
        vminspost = min(np.min(liste_sxmin_post), np.min(liste_symin_post))
        vmaxspost = max(np.max(liste_sxmax_post), np.max(liste_symax_post))

        C = C_2d_depuis_xi(xi, nu)
        transferer_C_2d(C, C_node, C_weights)
        K_rsi = lil_matrix((dof, dof))
        F_rsi = np.zeros(dof)
        B_template_rsi = np.zeros((3, 8))
        strain_elastic_rsi = []
        stress_rsi = []
        centers_rsi = []

        # --- Assemblage RSI avec quadrature de Gauss 2x2 ---
        gp = 1.0 / np.sqrt(3.0)
        gauss_pts = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]
        weights = [1.0, 1.0, 1.0, 1.0]

        def shape_dN_dxi(xy, eta):
            N = 0.25 * np.array([
                (1 - xy) * (1 - eta),
                (1 + xy) * (1 - eta),
                (1 + xy) * (1 + eta),
                (1 - xy) * (1 + eta)
            ])
            dN_dxi = 0.25 * np.array([
                [-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)],  # dN/dxi
                [-(1 - xy),  -(1 + xy),   (1 + xy),   (1 - xy)]   # dN/deta
            ])
            return N, dN_dxi

        for e, conn in enumerate(elements):
            xe_rsi = coords[conn]
            eps_inf = epsilon_inf[np.unravel_index(conn, (nny, nnx))]
            x_c_rsi = np.mean(xe_rsi, axis=0)
            deps_inf = np.mean(eps_inf)
            C_Utile_rsi = np.mean(C_node[np.unravel_index(conn, (nny, nnx))], axis=0)
            centers_rsi.append(x_c_rsi)

            Ke_rsi = np.zeros((8, 8))
            fe_rsi = np.zeros(8)

            # Intégration 2x2 de Gauss
            for (xy, eta), w in zip(gauss_pts, weights):
                N, dN_dxi = shape_dN_dxi(xy, eta)
                J_rsi = dN_dxi @ xe_rsi
                detJ_rsi = np.linalg.det(J_rsi)
                invJ_rsi = np.linalg.inv(J_rsi)
                dN_dx_rsi = invJ_rsi @ dN_dxi

                B_rsi = np.zeros((3, 8))
                for i in range(4):
                    B_rsi[0, 2*i  ] = dN_dx_rsi[0, i]
                    B_rsi[1, 2*i+1] = dN_dx_rsi[1, i]
                    B_rsi[2, 2*i  ] = dN_dx_rsi[1, i]
                    B_rsi[2, 2*i+1] = dN_dx_rsi[0, i]

                # Matrice de rigidité et vecteur élémentaire RSI
                Ke_rsi += (B_rsi.T @ C_Utile_rsi @ B_rsi) * detJ_rsi * w
                eps_rsi = (1 + nu) * deps_inf * np.array([1, 1, 0])  # deformation plane equivalente
                fe_rsi += (B_rsi.T @ C_Utile_rsi @ eps_rsi) * detJ_rsi * w

            # Assemblage global
            dof_indices_rsi = []
            for n in conn:
                dof_indices_rsi.extend([2*n, 2*n+1])

            for i in range(8):
                F_rsi[dof_indices_rsi[i]] += fe_rsi[i]
                for j in range(8):
                    K_rsi[dof_indices_rsi[i], dof_indices_rsi[j]] += Ke_rsi[i, j]

        # Conditions mecaniques RSI : celles definies APRES decoffrage.
        faces_rsi = (
            (clxinf2, [node_id(0,j) for j in range(nny)], 0),
            (clxsup2, [node_id(nnx-1,j) for j in range(nny)], 0),
            (clyinf2, [node_id(i,0) for i in range(nnx)], 1),
            (clysup2, [node_id(i,nny-1) for i in range(nnx)], 1),
        )
        bc_dofs_rsi = []
        for condition, nodes, normale in faces_rsi:
            if condition.lower() == 'symétrie':
                bc_dofs_rsi.extend(2*n + normale for n in nodes)
            elif condition != 'Bord libre':
                bc_dofs_rsi.extend(2*n + k for n in nodes for k in range(2))
        bc_dofs_rsi = completer_blocages_rigides(bc_dofs_rsi, coords, modes_rigides)
        appliquer_blocages_nuls(K_rsi, F_rsi, bc_dofs_rsi)

        K_rsi = csr_matrix(K_rsi)
        u_rsi = spsolve(K_rsi, F_rsi)

        for e, conn in enumerate(elements):
            eps_inf = epsilon_inf[np.unravel_index(conn, (nny, nnx))]
            xe_rsi = coords[conn]
            deps_inf = np.mean(eps_inf)
            ue_rsi = np.zeros(8)
            for i in range(4):
                ue_rsi[2*i] = u_rsi[2*conn[i]]
                ue_rsi[2*i+1] = u_rsi[2*conn[i]+1]
            J_rsi = precomputed_J
            invJ_rsi = np.linalg.inv(J_rsi)
            dN_dxi_rsi = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]) / 4
            dN_dx_rsi = invJ_rsi @ dN_dxi_rsi
            B_rsi = B_template_rsi.copy()
            for i in range(4):
                B_rsi[0, 2*i  ] = dN_dx_rsi[0, i]
                B_rsi[1, 2*i+1] = dN_dx_rsi[1, i]
                B_rsi[2, 2*i  ] = dN_dx_rsi[1, i]
                B_rsi[2, 2*i+1] = dN_dx_rsi[0, i]
            C_Utile_rsi = np.mean(C_node[np.unravel_index(conn, (nny, nnx))], axis=0)
            eps_total_rsi = B_rsi @ ue_rsi
            eps_rsi = (1 + nu) * deps_inf * np.array([1, 1, 0])  # deformation plane equivalente
            eps_el_rsi = eps_total_rsi - eps_rsi
            sigma_rsi = C_Utile_rsi @ eps_el_rsi
            # Conserver la deformation elastique physique dans le plan.
            strain_elastic_rsi.append(eps_total_rsi - deps_inf * np.array([1, 1, 0]))
            stress_rsi.append(sigma_rsi)

        eps_centre = epsilon_inf[nny//2, nnx//2]
        eps_max = float(np.max(epsilon_inf))
        u_x_rsi = u_rsi[0::2].reshape((nny, nnx))
        u_y_rsi = u_rsi[1::2].reshape((nny, nnx))
        s_xx_rsi = np.array([-s[0] for s in stress_rsi]).reshape((Ny, Nx))
        s_yy_rsi = np.array([-s[1] for s in stress_rsi]).reshape((Ny, Nx))

    else:
        # hx = 0.16
        # hy = 0.16
        # hz = 0.16
        Ny = int(largeur/hy)
        Nx = int(longueur/hx)
        Nz = int(hauteur/hz)
        nnx = Nx + 1
        nny = Ny + 1
        nnz = Nz + 1
        nn = nnx * nny * nnz
        ne = Nx * Ny * Nz
        coords = np.array([[i * hx, j * hy, k * hz] for k in range(nnz) for j in range(nny) for i in range(nnx)])
        dt = 0.1* min(hx**2, hy**2, hz**2) / D
        t = np.ones((Nz, Ny, Nx)) * T0
        xi = np.zeros((Nz, Ny, Nx))
        adexi = np.zeros((Nz, Ny, Nx))
        hxinf = get_h_value(clxinf)
        hxsup = get_h_value(clxsup)
        hyinf = get_h_value(clyinf)
        hysup = get_h_value(clysup)
        hzinf = get_h_value(clzinf)
        hzsup = get_h_value(clzsup)
        hxinf2 = get_h_value(clxinf2)
        hxsup2 = get_h_value(clxsup2)
        hyinf2 = get_h_value(clyinf2)
        hysup2 = get_h_value(clysup2)
        hzinf2 = get_h_value(clzinf2)
        hzsup2 = get_h_value(clzsup2)
        liste_temps = []
        liste_temps_meca = []
        liste_E = []
        liste_resbet = []
        liste_tmax = []
        liste_tcentre = []
        liste_uxmax_pre = []
        liste_uxmin_pre = []
        liste_uymin_pre = []
        liste_uymax_pre = []
        liste_uzmin_pre = []
        liste_uzmax_pre = []
        liste_sxmin_pre = []
        liste_symin_pre = []
        liste_sxmax_pre = []
        liste_symax_pre = []
        liste_szmin_pre = []
        liste_szmax_pre = []
        liste_uxmax_post = []
        liste_uxmin_post = []
        liste_uymin_post = []
        liste_uymax_post = []
        liste_uzmin_post = []
        liste_uzmax_post = []
        liste_sxmin_post = []
        liste_symin_post = []
        liste_sxmax_post = []
        liste_symax_post = []
        liste_szmin_post = []
        liste_szmax_post = []
        temperature_max = 0
        temps_temperature_max = 0
        abs_ux_max_pre = 0
        temps_abs_ux_max_pre = 0
        abs_uy_max_pre = 0
        temps_abs_uy_max_pre = 0
        abs_uz_max_pre = 0
        temps_abs_uz_max_pre = 0
        abs_sx_max_pre = 0
        temps_abs_sx_max_pre = 0
        abs_sy_max_pre = 0
        temps_abs_sy_max_pre = 0
        abs_sz_max_pre = 0
        temps_abs_sz_max_pre = 0
        abs_ux_max_post = 0
        temps_abs_ux_max_post = 0
        abs_uy_max_post = 0
        temps_abs_uy_max_post = 0
        abs_uz_max_post = 0
        temps_abs_uz_max_post = 0
        abs_sx_max_post = 0
        temps_abs_sx_max_post = 0
        abs_sy_max_post = 0
        temps_abs_sy_max_post = 0
        abs_sz_max_post = 0
        temps_abs_sz_max_post = 0
        grille_tmax_xy = np.zeros((Ny, Nx))
        grille_tmax_yz = np.zeros((Nz, Ny))
        grille_tmax_xz = np.zeros((Nz, Nx))
        # Champs complets necessaires aux cartes 3D meme si aucun maximum
        # n'a encore ete enregistre (ou si decoffrage apres la simulation).
        grille_uxmax_pre = np.zeros((nnz, nny, nnx))
        grille_uymax_pre = np.zeros((nnz, nny, nnx))
        grille_uzmax_pre = np.zeros((nnz, nny, nnx))
        grille_sxmax_pre = np.zeros((Nz, Ny, Nx))
        grille_symax_pre = np.zeros((Nz, Ny, Nx))
        grille_szmax_pre = np.zeros((Nz, Ny, Nx))
        grille_uxmax_post = np.zeros((nnz, nny, nnx))
        grille_uymax_post = np.zeros((nnz, nny, nnx))
        grille_uzmax_post = np.zeros((nnz, nny, nnx))
        grille_sxmax_post = np.zeros((Nz, Ny, Nx))
        grille_symax_post = np.zeros((Nz, Ny, Nx))
        grille_szmax_post = np.zeros((Nz, Ny, Nx))
        grille_uxmax_xy_pre = np.zeros((nny,nnx))
        grille_uxmax_yz_pre = np.zeros((nnz,nny))
        grille_uxmax_xz_pre = np.zeros((nnz,nnx))
        grille_uymax_xy_pre = np.zeros((nny,nnx))
        grille_uymax_yz_pre = np.zeros((nnz,nny))
        grille_uymax_xz_pre = np.zeros((nnz,nnx))
        grille_sxmax_xy_pre = np.zeros((nny,nnx))
        grille_sxmax_yz_pre = np.zeros((nnz,nny))
        grille_sxmax_xz_pre = np.zeros((nnz,nnx))
        grille_symax_xy_pre = np.zeros((nny,nnx))
        grille_symax_yz_pre = np.zeros((nnz,nny))
        grille_symax_xz_pre = np.zeros((nnz,nnx))
        grille_uxmax_xy_post = np.zeros((nny,nnx))
        grille_uxmax_yz_post = np.zeros((nnz,nny))
        grille_uxmax_xz_post = np.zeros((nnz,nnx))
        grille_uymax_xy_post = np.zeros((nny,nnx))
        grille_uymax_yz_post = np.zeros((nnz,nny))
        grille_uymax_xz_post = np.zeros((nnz,nnx))
        grille_sxmax_xy_post = np.zeros((nny,nnx))
        grille_sxmax_yz_post = np.zeros((nnz,nny))
        grille_sxmax_xz_post = np.zeros((nnz,nnx))
        grille_symax_xy_post = np.zeros((nny,nnx))
        grille_symax_yz_post = np.zeros((nnz,nny))
        grille_symax_xz_post = np.zeros((nnz,nnx))
        C = np.zeros((Nz, Ny, Nx, 6, 6))
        epsilon_inf = np.zeros((nnz, nny, nnx))
        u_x = np.zeros((nnz, nny, nnx))
        u_y = np.zeros((nnz, nny, nnx))
        s_xx = np.zeros((Nz, Ny, Nx))
        s_yy = np.zeros((Nz, Ny, Nx))
        T_node = np.zeros((nnz, nny, nnx))
        C_node = np.zeros((nnz, nny, nnx, 6, 6))
        T_weights = np.zeros((nnz, nny, nnx))
        #C_weights = np.zeros((nnz, nny, nnx, 6, 6))
        C_weights = np.zeros((nnz, nny, nnx))
        liste_epsilon_max = []
        liste_temps_epsilon_max = []

        def node_id(i, j, k):
            return k * nny * nnx + j * nnx + i

        def AffiniteChimique(ksi):
            hydrabis = np.array([0, 0.1091, 0.1259, 0.1259, 0.1301, 0.1371, 0.1483, 0.1538, 0.1636, 0.1734, 0.1832, 0.1930, 0.2028, 0.2126, 0.2224, 0.2322, 0.2420, 0.2518, 0.2616, 0.2714, 0.2812, 0.2910, 0.3008, 0.3106, 0.3203, 0.3301, 0.3399, 0.3497, 0.3595, 0.3693, 0.3791, 0.3889, 0.3987, 0.4085, 0.4183, 0.4281, 0.4477, 0.4673, 0.4869, 0.5065, 0.5261, 0.6855, 0.8183, 0.9752, 1])
            affchim = np.array([51.47, 51.47, 380.88, 439.22, 490.69, 545.59, 603.92, 665.69, 782.35, 840.69, 888.73, 933.33, 977.94, 1011.27, 1037.25, 1059.80, 1071.57, 1069.61, 1063.73, 1049.02, 1033.33, 1007.84, 978.43, 945.10, 909.80, 870.10, 832.35, 789.22, 746.08, 697.55, 650.98, 605.39, 560.29, 516.67, 473.04, 439.22, 368.14, 314.22, 265.20, 228.43, 187.75, 124.02, 67.16, 0, 0])
            result = np.zeros_like(ksi)
            mask = ksi < 1
            result[mask] = np.interp(ksi[mask], hydrabis, affchim) * 0.171
            return result

        elements = []
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    n1 = node_id(i, j, k)
                    n2 = node_id(i+1, j, k)
                    n3 = node_id(i+1, j+1, k)
                    n4 = node_id(i, j+1, k)
                    n5 = node_id(i, j, k+1)
                    n6 = node_id(i+1, j, k+1)
                    n7 = node_id(i+1, j+1, k+1)
                    n8 = node_id(i, j+1, k+1)
                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

        dof = 3 * nn
        start_time = time.time()
        iteration = 0

        # Pré-calculer certaines valeurs constantes
        precomputed_B_template = np.zeros((6, 24))
        precomputed_J = np.array([[hx/2, 0, 0], [0, hy/2, 0], [0, 0, hz/2]])
        detJ = np.linalg.det(precomputed_J)
        invJ = np.linalg.inv(precomputed_J)
        dN_dxi = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                            [-1, -1, 1, 1, -1, -1, 1, 1],
                            [-1, -1, -1, -1, 1, 1, 1, 1]]) / 8
        dN_dx = invJ @ dN_dxi
        coeff_diff_x = lam * dt / (rho * C_p * hx * hx)
        coeff_diff_y = lam * dt / (rho * C_p * hy * hy)
        coeff_diff_z = lam * dt / (rho * C_p * hz * hz)

        modes_rigides = base_mouvements_rigides(coords)
        while temps < duree and (np.max(t) - 273.15 > deltatmin or temps < 2*172800):
            etat_apres_decoffrage = (temps >= drc)
            cl_thermiques = ((hxinf2, hxsup2, hyinf2, hysup2, hzinf2, hzsup2)
                              if etat_apres_decoffrage
                              else (hxinf, hxsup, hyinf, hysup, hzinf, hzsup))
            t, xi = pas_thermique_3d(t, xi, hx, hy, hz, cl_thermiques, dt)

            Tmax = np.max(t) - 273.15
            T_centre = t[Nz//2, Ny//2, Nx//2] - 273.15

            if Tmax > 150:
                raise ValueError('Simulation interrompue : température > 150 °C.')

            if np.max(t) > temperature_max :
                grille_tmax = t.copy()
                temperature_max = np.max(t)
                temps_temperature_max = temps

            liste_temps.append(temps / 3600)
            liste_E.append(E(np.max(xi)) * 1e-9)
            liste_resbet.append(resbet((temps +1) / 3600))
            liste_tmax.append(Tmax)
            liste_tcentre.append(T_centre)

            T_node.fill(0)
            T_weights.fill(0)
            for dk in (0, 1):
                for dj in (0, 1):
                    for di in (0, 1):
                        T_node[dk:Nz+dk, dj:Ny+dj, di:Nx+di] += t
                        T_weights[dk:Nz+dk, dj:Ny+dj, di:Nx+di] += 1
            T_node /= T_weights

            # Création du masque des nœuds actifs
            mask = T_node > T_seuil
            # Calcul du terme d’évolution (uniquement sur les nœuds chauds)
            delta_epsilon = np.zeros_like(epsilon_inf)
            delta_epsilon[mask] = epsilon_m_point * dt * np.exp(-E_a_Diss / (R * (T_node[mask] - T_seuil)))
            # Mise à jour du champ
            epsilon_inf += delta_epsilon
            # Enregistrement des valeurs max et du temps courant
            liste_epsilon_max.append(np.max(epsilon_inf))
            liste_temps_epsilon_max.append(temps / 3600)

            if iteration % 100 == 0:

                liste_temps_meca.append(temps / 3600)

                C = C_3d_depuis_xi(xi)

                # Le champ thermique nodal T_node a deja ete calcule a ce pas.
                transferer_C_3d(C, C_node, C_weights)

                # Éléments finis, partie 1
                K = lil_matrix((dof, dof))
                F = np.zeros(dof)
                B_template = np.zeros((6, 24))
                strain_elastic = []
                stress = []
                centers = []

                gp = 1.0 / np.sqrt(3.0)
                gauss_pts = [
                    (-gp, -gp, -gp), ( gp, -gp, -gp),
                    ( gp,  gp, -gp), (-gp,  gp, -gp),
                    (-gp, -gp,  gp), ( gp, -gp,  gp),
                    ( gp,  gp,  gp), (-gp,  gp,  gp)
                ]
                weights = [1.0] * 8

                for e, conn in enumerate(elements):
                    xe = coords[conn]
                    Te = T_node[np.unravel_index(conn, (nnz, nny, nnx))]

                    idxs = [np.unravel_index(n, (nnz, nny, nnx)) for n in conn]
                    C_nodes_8 = np.array([C_node[k,j,i] for (k,j,i) in idxs])
                    C_Utile = np.mean(C_nodes_8, axis=0)

                    # C_Utile = np.mean(C_node[np.unravel_index(conn, (nnz, nny, nnx))], axis=0)
                    x_c = np.mean(xe, axis=0)
                    dT = np.mean(Te) - T0
                    centers.append(x_c)

                    Ke = np.zeros((24, 24))
                    fe_th = np.zeros(24)

                    for (xz, eta, zeta), w in zip(gauss_pts, weights):
                        # --- Fonctions de forme et dérivées par rapport à xi, eta, zeta ---
                        dN_dxi = 0.125 * np.array([
                            [-(1 - eta)*(1 - zeta),  (1 - eta)*(1 - zeta),  (1 + eta)*(1 - zeta), -(1 + eta)*(1 - zeta),
                            -(1 - eta)*(1 + zeta),  (1 - eta)*(1 + zeta),  (1 + eta)*(1 + zeta), -(1 + eta)*(1 + zeta)],
                            [-(1 - xz)*(1 - zeta),  -(1 + xz)*(1 - zeta),   (1 + xz)*(1 - zeta),   (1 - xz)*(1 - zeta),
                            -(1 - xz)*(1 + zeta),  -(1 + xz)*(1 + zeta),   (1 + xz)*(1 + zeta),   (1 - xz)*(1 + zeta)],
                            [-(1 - xz)*(1 - eta),   -(1 + xz)*(1 - eta),   -(1 + xz)*(1 + eta),   -(1 - xz)*(1 + eta),
                            (1 - xz)*(1 - eta),    (1 + xz)*(1 - eta),    (1 + xz)*(1 + eta),    (1 - xz)*(1 + eta)]
                        ])

                        # --- Jacobien et dérivées spatiales ---
                        J = dN_dxi @ xe
                        detJ = np.linalg.det(J)
                        invJ = np.linalg.inv(J)
                        dN_dx = invJ @ dN_dxi

                        # --- Construction de la matrice B ---
                        B = np.zeros((6, 24))
                        for i in range(8):
                            dNi_dx = dN_dx[0, i]
                            dNi_dy = dN_dx[1, i]
                            dNi_dz = dN_dx[2, i]
                            B[0, 3*i  ] = dNi_dx
                            B[1, 3*i+1] = dNi_dy
                            B[2, 3*i+2] = dNi_dz
                            B[3, 3*i  ] = dNi_dy
                            B[3, 3*i+1] = dNi_dx
                            B[4, 3*i+1] = dNi_dz
                            B[4, 3*i+2] = dNi_dy
                            B[5, 3*i  ] = dNi_dz
                            B[5, 3*i+2] = dNi_dx

                        # --- Matrice de rigidité élémentaire au point de Gauss ---
                        Ke += (B.T @ C_Utile @ B) * detJ * w

                        # --- Effort thermique équivalent ---
                        eps_th = alpha * dT * np.array([1, 1, 1, 0, 0, 0])
                        fe_th += (B.T @ C_Utile @ eps_th) * detJ * w

                    # --- Assemblage global ---
                    dof_indices = []
                    for n in conn:
                        dof_indices.extend([3*n, 3*n+1, 3*n+2])
                    for i in range(24):
                        F[dof_indices[i]] += fe_th[i] #* hx * hy * hz / 8.0
                        for j in range(24):
                            K[dof_indices[i], dof_indices[j]] += Ke[i, j] #* hx * hy * hz / 8.0

                # Conditions MECANIQUES : ne pas muter h (coef. d'echange) dans la mecanique.
                conds = ((clxinf2, clxsup2, clyinf2, clysup2, clzinf2, clzsup2)
                         if temps >= drc else
                         (clxinf, clxsup, clyinf, clysup, clzinf, clzsup))
                faces = (
                    (conds[0], [node_id(0,j,k) for k in range(nnz) for j in range(nny)], 0),
                    (conds[1], [node_id(nnx-1,j,k) for k in range(nnz) for j in range(nny)], 0),
                    (conds[2], [node_id(i,0,k) for k in range(nnz) for i in range(nnx)], 1),
                    (conds[3], [node_id(i,nny-1,k) for k in range(nnz) for i in range(nnx)], 1),
                    (conds[4], [node_id(i,j,0) for j in range(nny) for i in range(nnx)], 2),
                    (conds[5], [node_id(i,j,nnz-1) for j in range(nny) for i in range(nnx)], 2),
                )
                bc_dofs = []
                for condition, nodes, normale in faces:
                    if condition.lower() == 'symétrie':
                        bc_dofs.extend(3*n + normale for n in nodes)
                    elif condition != 'Bord libre':
                        bc_dofs.extend(3*n + k for n in nodes for k in range(3))
                bc_dofs = completer_blocages_rigides(bc_dofs, coords, modes_rigides)
                appliquer_blocages_nuls(K, F, bc_dofs)

                # Fin des conditions aux limites à modifier
                # Éléments finis, partie 2
                K = csr_matrix(K)
                u = spsolve(K, F)

                for e, conn in enumerate(elements):
                    Te = T_node[np.unravel_index(conn, (nnz, nny, nnx))]
                    xe = coords[conn]
                    dT = np.mean(Te) - T0
                    ue = np.zeros(24)
                    for i in range(8):
                        ue[3*i] = u[3*conn[i]]
                        ue[3*i+1] = u[3*conn[i]+1]
                        ue[3*i+2] = u[3*conn[i]+2]
                    J = precomputed_J
                    invJ = np.linalg.inv(J)
                    dN_dxi = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                                        [-1, -1, 1, 1, -1, -1, 1, 1],
                                        [-1, -1, -1, -1, 1, 1, 1, 1]]) / 8
                    dN_dx = invJ @ dN_dxi
                    B = B_template.copy()
                    for i in range(8):
                        B[0, 3*i] = dN_dx[0, i]
                        B[1, 3*i+1] = dN_dx[1, i]
                        B[2, 3*i+2] = dN_dx[2, i]
                        B[3, 3*i] = dN_dx[1, i]
                        B[3, 3*i+1] = dN_dx[0, i]
                        B[4, 3*i+1] = dN_dx[2, i]
                        B[4, 3*i+2] = dN_dx[1, i]
                        B[5, 3*i] = dN_dx[2, i]
                        B[5, 3*i+2] = dN_dx[0, i]
                    C_Utile = np.mean(C_node[np.unravel_index(conn, (nnz, nny, nnx))], axis=0)
                    eps_total = B @ ue
                    eps_th = alpha * dT * np.array([1, 1, 1, 0, 0, 0])
                    eps_el = eps_total - eps_th
                    sigma = C_Utile @ eps_el
                    strain_elastic.append(eps_el)
                    stress.append(sigma)

                u_x = u[0::3].reshape((nnz, nny, nnx))
                u_y = u[1::3].reshape((nnz, nny, nnx))
                u_z = u[2::3].reshape((nnz, nny, nnx))

                s_xx = np.array([-s[0] for s in stress]).reshape((Nz, Ny, Nx))
                s_yy = np.array([-s[1] for s in stress]).reshape((Nz, Ny, Nx))
                s_zz = np.array([-s[2] for s in stress]).reshape((Nz, Ny, Nx))

                #print(temps/3600)
                liste_uxmax_pre.append(np.max(u_x[1:-1,1:-1,1:-1]))
                liste_uxmin_pre.append(np.min(u_x[1:-1,1:-1,1:-1]))
                liste_uymin_pre.append(np.min(u_y[1:-1,1:-1,1:-1]))
                liste_uymax_pre.append(np.max(u_y[1:-1,1:-1,1:-1]))
                liste_uzmin_pre.append(np.min(u_z[1:-1,1:-1,1:-1]))
                liste_uzmax_pre.append(np.max(u_z[1:-1,1:-1,1:-1]))
                liste_sxmin_pre.append(np.min(s_xx[1:-1,1:-1,1:-1]))
                liste_symin_pre.append(np.min(s_yy[1:-1,1:-1,1:-1]))
                liste_sxmax_pre.append(np.max(s_xx[1:-1,1:-1,1:-1]))
                liste_symax_pre.append(np.max(s_yy[1:-1,1:-1,1:-1]))
                liste_szmin_pre.append(np.min(s_zz[1:-1,1:-1,1:-1]))
                liste_szmax_pre.append(np.max(s_zz[1:-1,1:-1,1:-1]))
                if temps > drc :
                    liste_uxmax_post.append(np.max(u_x[1:-1,1:-1,1:-1]))
                    liste_uxmin_post.append(np.min(u_x[1:-1,1:-1,1:-1]))
                    liste_uymin_post.append(np.min(u_y[1:-1,1:-1,1:-1]))
                    liste_uymax_post.append(np.max(u_y[1:-1,1:-1,1:-1]))
                    liste_uzmin_post.append(np.min(u_z[1:-1,1:-1,1:-1]))
                    liste_uzmax_post.append(np.max(u_z[1:-1,1:-1,1:-1]))
                    liste_sxmin_post.append(np.min(s_xx[1:-1,1:-1,1:-1]))
                    liste_symin_post.append(np.min(s_yy[1:-1,1:-1,1:-1]))
                    liste_sxmax_post.append(np.max(s_xx[1:-1,1:-1,1:-1]))
                    liste_symax_post.append(np.max(s_yy[1:-1,1:-1,1:-1]))
                    liste_szmin_post.append(np.min(s_zz[1:-1,1:-1,1:-1]))
                    liste_szmax_post.append(np.max(s_zz[1:-1,1:-1,1:-1]))
                else :
                    liste_uxmax_post.append(0)
                    liste_uxmin_post.append(0)
                    liste_uymin_post.append(0)
                    liste_uymax_post.append(0)
                    liste_uzmin_post.append(0)
                    liste_uzmax_post.append(0)
                    liste_sxmin_post.append(0)
                    liste_symin_post.append(0)
                    liste_sxmax_post.append(0)
                    liste_symax_post.append(0)
                    liste_szmin_post.append(0)
                    liste_szmax_post.append(0)

                if np.max(np.abs(u_x[1:-1,1:-1,1:-1])) > abs_ux_max_pre:
                    grille_uxmax_pre = u_x.copy()
                    abs_ux_max_pre = np.max(np.abs(u_x[1:-1,1:-1,1:-1]))
                    temps_abs_ux_max_pre = temps
                if np.max(np.abs(u_y[1:-1,1:-1,1:-1])) > abs_uy_max_pre:
                    grille_uymax_pre = u_y.copy()
                    abs_uy_max_pre = np.max(np.abs(u_y[1:-1,1:-1,1:-1]))
                    temps_abs_uy_max_pre = temps
                if np.max(np.abs(u_z[1:-1,1:-1,1:-1])) > abs_uz_max_pre:
                    grille_uzmax_pre = u_z.copy()
                    abs_uz_max_pre = np.max(np.abs(u_z[1:-1,1:-1,1:-1]))
                    temps_abs_uz_max_pre = temps
                if np.max(np.abs(s_xx[1:-1,1:-1,1:-1])) > abs_sx_max_pre:
                    grille_sxmax_pre = s_xx.copy()
                    abs_sx_max_pre = np.max(np.abs(s_xx[1:-1,1:-1,1:-1]))
                    temps_abs_sx_max_pre = temps
                if np.max(np.abs(s_yy[1:-1,1:-1,1:-1])) > abs_sy_max_pre:
                    grille_symax_pre = s_yy.copy()
                    abs_sy_max_pre = np.max(np.abs(s_yy[1:-1,1:-1,1:-1]))
                    temps_abs_sy_max_pre = temps
                if np.max(np.abs(s_zz[1:-1,1:-1,1:-1])) > abs_sz_max_pre:
                    grille_szmax_pre = s_zz.copy()
                    abs_sz_max_pre = np.max(np.abs(s_zz[1:-1,1:-1,1:-1]))
                    temps_abs_sz_max_pre = temps
                if temps > drc :
                    if np.max(np.abs(u_x[1:-1,1:-1,1:-1])) > abs_ux_max_post:
                        grille_uxmax_post = u_x.copy()
                        abs_ux_max_post = np.max(np.abs(u_x[1:-1,1:-1,1:-1]))
                        temps_abs_ux_max_post = temps
                    if np.max(np.abs(u_y[1:-1,1:-1,1:-1])) > abs_uy_max_post:
                        grille_uymax_post = u_y.copy()
                        abs_uy_max_post = np.max(np.abs(u_y[1:-1,1:-1,1:-1]))
                        temps_abs_uy_max_post = temps
                    if np.max(np.abs(u_z[1:-1,1:-1,1:-1])) > abs_uz_max_post:
                        grille_uzmax_post = u_z.copy()
                        abs_uz_max_post = np.max(np.abs(u_z[1:-1,1:-1,1:-1]))
                        temps_abs_uz_max_post = temps
                    if np.max(np.abs(s_xx[1:-1,1:-1,1:-1])) > abs_sx_max_post:
                        grille_sxmax_post = s_xx.copy()
                        abs_sx_max_post = np.max(np.abs(s_xx[1:-1,1:-1,1:-1]))
                        temps_abs_sx_max_post = temps
                    if np.max(np.abs(s_yy[1:-1,1:-1,1:-1])) > abs_sy_max_post:
                        grille_symax_post = s_yy.copy()
                        abs_sy_max_post = np.max(np.abs(s_yy[1:-1,1:-1,1:-1]))
                        temps_abs_sy_max_post = temps
                    if np.max(np.abs(s_zz[1:-1,1:-1,1:-1])) > abs_sz_max_post:
                        grille_szmax_post = s_zz.copy()
                        abs_sz_max_post = np.max(np.abs(s_zz[1:-1,1:-1,1:-1]))
                        temps_abs_sz_max_post = temps

            temps += dt
            iteration += 1


        vminupre = min(np.min(liste_uxmin_pre), np.min(liste_uymin_pre), np.min(liste_uzmin_pre))
        vmaxupre = max(np.max(liste_uxmax_pre), np.max(liste_uymax_pre), np.max(liste_uzmax_pre))
        vminspre = min(np.min(liste_sxmin_pre), np.min(liste_symin_pre), np.min(liste_szmin_pre))
        vmaxspre = max(np.max(liste_sxmax_pre), np.max(liste_symax_pre), np.max(liste_szmax_pre))
        vminupost = min(np.min(liste_uxmin_post), np.min(liste_uymin_post), np.min(liste_uzmin_post))
        vmaxupost = max(np.max(liste_uxmax_post), np.max(liste_uymax_post), np.max(liste_uzmax_post))
        vminspost = min(np.min(liste_sxmin_post), np.min(liste_symin_post), np.min(liste_szmin_post))
        vmaxspost = max(np.max(liste_sxmax_post), np.max(liste_symax_post), np.max(liste_szmax_post))


        C = C_3d_depuis_xi(xi)
        transferer_C_3d(C, C_node, C_weights)
        K_rsi = lil_matrix((dof, dof))
        F_rsi = np.zeros(dof)
        B_template_rsi = np.zeros((6, 24))
        strain_elastic_rsi = []
        stress_rsi = []
        centers_rsi = []

        gp = 1.0 / np.sqrt(3.0)
        gauss_pts = [
            (-gp, -gp, -gp), ( gp, -gp, -gp),
            ( gp,  gp, -gp), (-gp,  gp, -gp),
            (-gp, -gp,  gp), ( gp, -gp,  gp),
            ( gp,  gp,  gp), (-gp,  gp,  gp)
        ]
        weights = [1.0] * 8

        for e, conn in enumerate(elements):
            xe_rsi = coords[conn]
            eps_inf = epsilon_inf[np.unravel_index(conn, (nnz, nny, nnx))]
            x_c_rsi = np.mean(xe_rsi, axis=0)
            deps_inf = np.mean(eps_inf)
            # C_Utile_rsi = np.mean(C_node[np.unravel_index(conn, (nnz, nny, nnx))], axis=0)
            idxs = [np.unravel_index(n, (nnz, nny, nnx)) for n in conn]
            C_nodes_8 = np.array([C_node[k,j,i] for (k,j,i) in idxs])
            C_Utile_rsi = np.mean(C_nodes_8, axis=0)

            centers_rsi.append(x_c_rsi)

            Ke_rsi = np.zeros((24, 24))
            fe_rsi = np.zeros(24)

            # --- Boucle sur les 8 points de Gauss ---
            for (xn, eta, zeta), w in zip(gauss_pts, weights):
                # Dérivées locales des fonctions de forme
                dN_dxi_rsi = 0.125 * np.array([
                    [-(1 - eta)*(1 - zeta),  (1 - eta)*(1 - zeta),  (1 + eta)*(1 - zeta), -(1 + eta)*(1 - zeta),
                    -(1 - eta)*(1 + zeta),  (1 - eta)*(1 + zeta),  (1 + eta)*(1 + zeta), -(1 + eta)*(1 + zeta)],
                    [-(1 - xn)*(1 - zeta),  -(1 + xn)*(1 - zeta),   (1 + xn)*(1 - zeta),   (1 - xn)*(1 - zeta),
                    -(1 - xn)*(1 + zeta),  -(1 + xn)*(1 + zeta),   (1 + xn)*(1 + zeta),   (1 - xn)*(1 + zeta)],
                    [-(1 - xn)*(1 - eta),   -(1 + xn)*(1 - eta),   -(1 + xn)*(1 + eta),   -(1 - xn)*(1 + eta),
                    (1 - xn)*(1 - eta),    (1 + xn)*(1 - eta),    (1 + xn)*(1 + eta),    (1 - xn)*(1 + eta)]
                ])

                # --- Jacobien et dérivées spatiales ---
                J_rsi = dN_dxi_rsi @ xe_rsi
                detJ_rsi = np.linalg.det(J_rsi)
                invJ_rsi = np.linalg.inv(J_rsi)
                dN_dx_rsi = invJ_rsi @ dN_dxi_rsi

                # --- Construction de la matrice B ---
                B_rsi = np.zeros((6, 24))
                for i in range(8):
                    dNi_dx = dN_dx_rsi[0, i]
                    dNi_dy = dN_dx_rsi[1, i]
                    dNi_dz = dN_dx_rsi[2, i]
                    B_rsi[0, 3*i  ] = dNi_dx
                    B_rsi[1, 3*i+1] = dNi_dy
                    B_rsi[2, 3*i+2] = dNi_dz
                    B_rsi[3, 3*i  ] = dNi_dy
                    B_rsi[3, 3*i+1] = dNi_dx
                    B_rsi[4, 3*i+1] = dNi_dz
                    B_rsi[4, 3*i+2] = dNi_dy
                    B_rsi[5, 3*i  ] = dNi_dz
                    B_rsi[5, 3*i+2] = dNi_dx

                # --- Intégration au point de Gauss ---
                Ke_rsi += (B_rsi.T @ C_Utile_rsi @ B_rsi) * detJ_rsi * w
                eps_rsi = deps_inf * np.array([1, 1, 1, 0, 0, 0])
                fe_rsi += (B_rsi.T @ C_Utile_rsi @ eps_rsi) * detJ_rsi * w

            # --- Assemblage global ---
            dof_indices_rsi = []
            for n in conn:
                dof_indices_rsi.extend([3*n, 3*n+1, 3*n+2])
            for i in range(24):
                F_rsi[dof_indices_rsi[i]] += fe_rsi[i] #* hx * hy * hz / 8.0
                for j in range(24):
                    K_rsi[dof_indices_rsi[i], dof_indices_rsi[j]] += Ke_rsi[i, j] #* hx * hy * hz / 8.0


        faces_rsi = (
            (clxinf2, [node_id(0,j,k) for k in range(nnz) for j in range(nny)], 0),
            (clxsup2, [node_id(nnx-1,j,k) for k in range(nnz) for j in range(nny)], 0),
            (clyinf2, [node_id(i,0,k) for k in range(nnz) for i in range(nnx)], 1),
            (clysup2, [node_id(i,nny-1,k) for k in range(nnz) for i in range(nnx)], 1),
            (clzinf2, [node_id(i,j,0) for j in range(nny) for i in range(nnx)], 2),
            (clzsup2, [node_id(i,j,nnz-1) for j in range(nny) for i in range(nnx)], 2),
        )
        bc_dofs_rsi = []
        for condition, nodes, normale in faces_rsi:
            if condition.lower() == 'symétrie':
                bc_dofs_rsi.extend(3*n + normale for n in nodes)
            elif condition != 'Bord libre':
                bc_dofs_rsi.extend(3*n + k for n in nodes for k in range(3))
        bc_dofs_rsi = completer_blocages_rigides(bc_dofs_rsi, coords, modes_rigides)
        appliquer_blocages_nuls(K_rsi, F_rsi, bc_dofs_rsi)

        K_rsi = csr_matrix(K_rsi)
        u_rsi = spsolve(K_rsi, F_rsi)

        for e, conn in enumerate(elements):
            eps_inf = epsilon_inf[np.unravel_index(conn, (nnz, nny, nnx))]
            xe_rsi = coords[conn]
            deps_inf = np.mean(eps_inf)
            ue_rsi = np.zeros(24)
            for i in range(8):
                ue_rsi[3*i] = u_rsi[3*conn[i]]
                ue_rsi[3*i+1] = u_rsi[3*conn[i]+1]
                ue_rsi[3*i+2] = u_rsi[3*conn[i]+2]
            J_rsi = precomputed_J
            invJ_rsi = np.linalg.inv(J_rsi)
            dN_dxi_rsi = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                                    [-1, -1, 1, 1, -1, -1, 1, 1],
                                    [-1, -1, -1, -1, 1, 1, 1, 1]]) / 8
            dN_dx_rsi = invJ_rsi @ dN_dxi_rsi
            B_rsi = B_template_rsi.copy()
            for i in range(8):
                B_rsi[0, 3*i] = dN_dx_rsi[0, i]
                B_rsi[1, 3*i+1] = dN_dx_rsi[1, i]
                B_rsi[2, 3*i+2] = dN_dx_rsi[2, i]
                B_rsi[3, 3*i] = dN_dx_rsi[1, i]
                B_rsi[3, 3*i+1] = dN_dx_rsi[0, i]
                B_rsi[4, 3*i+1] = dN_dx_rsi[2, i]
                B_rsi[4, 3*i+2] = dN_dx_rsi[1, i]
                B_rsi[5, 3*i] = dN_dx_rsi[2, i]
                B_rsi[5, 3*i+2] = dN_dx_rsi[0, i]
            C_Utile_rsi = np.mean(C_node[np.unravel_index(conn, (nnz, nny, nnx))], axis=0)
            eps_total_rsi = B_rsi @ ue_rsi
            eps_rsi = deps_inf * np.array([1, 1, 1, 0, 0, 0])
            eps_el_rsi = eps_total_rsi - eps_rsi
            sigma_rsi = C_Utile_rsi @ eps_el_rsi
            strain_elastic_rsi.append(eps_el_rsi)
            stress_rsi.append(sigma_rsi)

        eps_centre = epsilon_inf[nnz//2, nny//2, nnx//2]
        eps_max = float(np.max(epsilon_inf))
        u_x_rsi = u_rsi[0::3].reshape((nnz, nny, nnx))
        u_y_rsi = u_rsi[1::3].reshape((nnz, nny, nnx))
        u_z_rsi = u_rsi[2::3].reshape((nnz, nny, nnx))

        s_xx_rsi = np.array([-s[0] for s in stress_rsi]).reshape((Nz, Ny, Nx))
        s_yy_rsi = np.array([-s[1] for s in stress_rsi]).reshape((Nz, Ny, Nx))
        s_zz_rsi = np.array([-s[2] for s in stress_rsi]).reshape((Nz, Ny, Nx))

    data = locals()
    def arr(key, multiplier=1.0, offset=0.0):
        return (np.asarray(data[key], dtype=float) * multiplier + offset).copy()
    fields = {
        'Température au pic thermique': {'data': arr('grille_tmax', offset=-273.15), 'unit':'°C', 'position':'cell'},
        'Température finale': {'data': arr('t', offset=-273.15), 'unit':'°C', 'position':'cell'},
        'Degré d’hydratation final': {'data': arr('xi'), 'unit':'–', 'position':'cell'},
        'Déformation RSI ultime': {'data': arr('epsilon_inf', multiplier=100), 'unit':'%', 'position':'point'},
    }
    for axis in ('x','y','z')[:dim]:
        for phase, name in [('pre', 'Jeune âge – tous instants'), ('post', 'Jeune âge – après décoffrage')]:
            k=f'grille_u{axis}max_{phase}'
            sk=f'grille_s{axis}max_{phase}'
            if k in data:
                fields[f'Déplacement U{axis} ({name})']={'data':arr(k,1e3),'unit':'mm','position':'point'}
            if sk in data:
                fields[f'Contrainte σ{axis}{axis} ({name})']={'data':arr(sk,1e-6),'unit':'MPa','position':'cell'}
        fields[f'Déplacement RSI U{axis}']={'data':arr(f'u_{axis}_rsi',1e3),'unit':'mm','position':'point'}
        fields[f'Contrainte RSI σ{axis}{axis}']={'data':arr(f's_{axis}{axis}_rsi',1e-6),'unit':'MPa','position':'cell'}
    result={
        'dimension':dim,
        'dimensions_m':[longueur,largeur]+([hauteur] if dim==3 else []),
        'pas_m':[hx,hy]+([hz] if dim==3 else []),
        'dt_s':dt,
        'run_time_s':time.time()-start_time,
        'times_h':arr('liste_temps'),
        'times_meca_h':arr('liste_temps_meca'),
        'tmax_C':arr('liste_tmax'),
        'tcentre_C':arr('liste_tcentre'),
        'young_GPa':arr('liste_E'),
        'resistance_MPa':arr('liste_resbet'),
        'times_epsilon_h':arr('liste_temps_epsilon_max'),
        'epsilon_max_percent':arr('liste_epsilon_max',100),
        'peak_temperature_C':float(np.max(grille_tmax)-273.15),
        'peak_temperature_time_h':float(temps_temperature_max/3600),
        'max_epsilon_percent':float(eps_max*100),
        'fields':fields,
        'params':dict(p),
    }
    if progress_callback:
        progress_callback(1.0)
    return result
