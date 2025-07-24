from ncempy.io import dm
import numpy as np
import paramiko
import time, os
import h5py

# Class for listening and logging new files
class logger:
    """Used to open directory to listen to
    Logs the currently used images
    Logs most recent tilt list
    Logs most recent image array
    ???Logs most recent reconstruction???
    """
    def __init__(self,listenDirectory,fileExtension):

        # Main listening variables
        if not os.path.exists(listenDirectory):
            os.makedirs(listenDirectory)

        self.listenDir = listenDirectory
        self.fileExt = fileExtension
        self.log_dir = self.access_log_directory(self.listenDir)
        (self.log_projs,self.log_tilts) = self.access_tilt_projs_H5(self.log_dir)
        self.log_files = self.access_files_list(self.log_dir)
        self.files_path = '/'.join([self.log_dir, "used.txt"])
        self.listen_files = self.listen_files_list(self.listenDir)
        print("Listener on {} created.".format(self.listenDir))

        # SFTP variables
        self.sftp_connection = []

    # SFTP variables
    def set_login_credentials(self, hostName, userName, password, port):
        self.myHostname = hostName
        self.myUsername = userName
        self.myPassword = password
        self.myPort = port

        # Whether to Monitor the Data Locally or From Server
        if userName != 'localhost': self.begin_sftp(self.remoteDir)
        else: self.monitor_local()

    def access_log_directory(self,directory):
        """Access logdirectory or create if needed"""
        logDirectory = directory +"/log"
        if not os.path.exists(logDirectory):
            os.makedirs(logDirectory)

        return logDirectory

    def access_tilt_projs_H5(self,directory):
        h5_path = "{}/projs_tilts.h5".format(directory)

        if not os.path.exists(h5_path):
            self.log_projs, self.log_tilts = [], []
        else:
            h5 = h5py.File(h5_path,'r')
            self.log_tilts = h5["tilts"][:]
            self.log_projs = h5["projs"][:,:,:]
            h5.close()
            return (projs,tilts)

    def access_files_list(self,directory):
        """Inside given directory, access files list (.txt)"""
        projPath = "used.txt"
        totalPath = '/'.join([directory,projPath])
        if not os.path.exists(totalPath):
            return []
        else:
            with open(totalPath,"r") as f:
                for line in f:
                    return line.split(",")

    def listen_files_list(self,directory):
        """Grab current files in listen directory"""
        #files = glob.glob('/'.join((directory,'*.dm4')))
        files = [f for f in os.listdir(directory) if f[-len(self.fileExt):] == self.fileExt]
        files.sort(key = lambda x: x[:3])
        return files

    def CHANGE_appendAll(self):
        """Append stored files for each new element"""
        # Separate new files to be loaded
        FoI = list(set(self.listen_files)-set(self.log_files))
        FoI.sort(key = lambda x: x[:3])
        for file in FoI:
            print("Loading {}".format(file))
            file_path = "{}/{}".format(self.listenDir,file)

            try: 
                dmRef = dm.dmReader(file_path)
                newProj = self.center_of_mass_align(self.background_subtract(dmRef['data']))
                newAngle = self.acquire_proj_angle(file_path)

                self.log_tilts = np.append(self.log_tilts,newAngle)
                self.log_files = np.append(self.log_files,file)

                # Account for Python's disdain for AxAx1 arrays (compresses to 2D)
                if(len(self.log_projs) == 0):
                    dataDim = np.shape(newProj)
                    self.log_projs = np.zeros([dataDim[0],dataDim[1],1])
                    self.log_projs[:,:,0] = newProj
                else:
                    self.log_projs = np.dstack((self.log_projs,newProj))

            except:
                print('Could not read : {}, will preceed with reconstruction and re-download on next pass'.format(file))
                break

        self.CHANGE_saveAll()

    def CHANGE_saveAll(self):
        """Save all log data held"""

        self.h5_path = '{}/projs_tilts.h5'.format(self.log_dir)
        h5=h5py.File(self.h5_path, 'w')
        h5.create_dataset('tilts',data = self.log_tilts)
        h5.create_dataset('projs',data = self.log_projs)
        h5.close()

        with open(self.files_path, "w") as f:
            f.write(",".join(self.log_files))

    def monitor_local(self,seconds=1):
        """Return true if, within seconds, any new files exist in listenDir
        NOTE: Lazy Scheme is used (set difference), can update """

        for ts in range(0,seconds):
            self.listen_files = self.listen_files_list(self.listenDir)
            FoI = list(set(self.listen_files)-set(self.log_files))
            if len(FoI) == 0:
                time.sleep(1)
            else:
                self.CHANGE_appendAll()
                return True;

        return False

    def monitor_online(self,seconds=1):
        """Return true if, within seconds, any new .dm4 exist in remote_dir
        REQUIRES: begin_sftp has been called with appropriate directory
        NOTE: Lazy Scheme is used (set difference), can update """
        for ts in range(0,seconds):

            # Read-in only .dm4 files for tracking changes
            remote_filesList = [ f for f in self.sftp_connection.listdir() if f[-len(self.fileExt):] == self.fileExt]
            remote_filesList.sort(key = lambda x: x[:3])
            FoI = list(set(remote_filesList)-set(self.log_files))

            if len(FoI) == 0:
                time.sleep(1)
            else:
                for file in FoI:
                    # Ensure local copy of data for future use
                    self.sftp_connection.get(file,"{}/{}".format(self.listenDir,file))

                # New list of local .dm4 to observe
                self.listen_files = self.listen_files_list(self.listenDir)
                self.CHANGE_appendAll()
                return True

        return False

    def begin_sftp(self,remoteDirectory):

        self.transport = paramiko.Transport((self.myHostname, self.myPort))
        self.transport.connect(username=self.myUsername,password=self.myPassword)
        self.sftp_connection = paramiko.SFTPClient.from_transport(self.transport)

        self.remote_dir = remoteDirectory
        self.sftp_connection.chdir(remoteDirectory)

        self.monitor_local()
        self.monitor_online()

    def acquire_proj_angle(self,file):
        """Acquires angles from metadata of .dm4 files"""
        with dm.fileDM(file) as inDM:
            alphaTag = ".ImageList.2.ImageTags.Microscope Info.Stage Position.Stage Alpha"
            return inDM.allTags[alphaTag]


    def load_results(self, fname, tomo):
        fullFname = '{}/{}.h5'.format(self.listenDir,fname)
        if os.path.exists(fullFname):
            print('Previous checkpoint found at: ', fullFname)

            h5=h5py.File(fullFname,'r')

            # Load Recon to tomo
            recon = h5['recon'][:,:,:]
            for s in range(tomo.NsliceLoc()):
                tomo.setRecon(recon[s+tomo.firstSlice(),:,:],s)

            # Try to load previous dd and tv if available
            try: dd = h5['results/fullDD'][:]
            except: dd = np.array([])
            
            try: tv = h5['results/fullTV'][:]
            except: tv = np.array([]) 
            h5.close()

            return (dd, tv)
        else:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def load_tilt_series(self, tomo):

        (Nslice, Nray, Nproj) = self.log_projs.shape
        b = np.zeros([Nslice, Nray * Nproj], dtype=np.float32)
        for s in range(Nslice):
            b[s,:] = self.log_projs[s,:,:].transpose().ravel()
        tomo.setTiltSeries(b)

    def save_results(self, fname, tomo, meta=None, results=None):

        fullFname = '{}/{}.h5'.format(self.listenDir,fname)

        tomo.saveRecon(fullFname, 0)

        h5=h5py.File(fullFname, 'a')

        if meta != None:
            params = h5.create_group("parameters")
            for key,item in meta.items():
                params.attrs[key] = item

        if results != None:
            conv = h5.create_group("results")
            for key,item in results.items():
                conv.create_dataset(key, dtype=np.float32, data=item)
        h5.close()

    # Shift Image so that the center of mass is at the origin"
    # Automatically align tilt images by center of mass method
    def center_of_mass_align(self, image):
        (Nx, Ny) = image.shape
        y = np.linspace(0,Ny-1,Ny)
        x = np.linspace(0,Nx-1,Nx)
        [X, Y] = np.meshgrid(x, y, indexing="ij")

        imageCOM_x = int(np.sum(image * X) / np.sum(image))
        imageCOM_y = int(np.sum(image * Y) / np.sum(image))

        sx = -(imageCOM_x - Nx // 2)
        sy = -(imageCOM_y - Ny // 2)

        output = np.roll(image, sx, axis=0)
        output = np.roll(output, sy, axis=1)

        return output

    # Remove Background Intensity
    def background_subtract(self, image):

        (Nx, Ny) = image.shape
        XRANGE = np.array([0, int(Nx//4)], dtype=int)
        YRANGE = np.array([0, int(Ny//4)], dtype=int)

        image -= np.average(image[XRANGE[0]:XRANGE[1], YRANGE[0]:YRANGE[1]])

        return image
