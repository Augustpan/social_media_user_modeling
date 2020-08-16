import pandas

class SMUM_Data: # Social Media User Modeling Data

    def __init__(self):
        # Class member variables are defined below
        # Pandas DataFrame instances
        self.checkins   = []
        self.profile    = []
        self.social     = []
        self.tags       = []
        self.tweets     = []

    def load(self):
        # Utility function
        def _eval(x):
            if x == 'NULL': return None
            else:           return eval(x)

        # Load location data from checkins.txt
        with open('data/checkins.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf-8')
            lst = line.split('\t')
            if len(lst) != 8:
                lst.append('')
            par = (int(lst[0]),     # def: User ID,     type: int
                   lst[1],          # def: POI,         type: str
                   _eval(lst[2]),   # def: Category1,   type: None or int
                   _eval(lst[3]),   # def: Category2,   type: None or int
                   _eval(lst[4]),   # def: Category3,   type: None or int
                   _eval(lst[5]),   # def: Latitude,    type: float
                   _eval(lst[6]),   # def: Longitude,   type: float
                   lst[7])          # def: Name,        type: str
            self.checkins.append(par)
        colum_tag = ['id','poi','cat1','cat2','cat3','lat','lng','name']
        self.checkins = pandas.DataFrame(self.checkins, columns=colum_tag)
        print '<checkins data loaded, total record ' + str(len(self.checkins.index)) + '>'
        # Load user's profile(gender) from profile.txt
        with open('data/profile.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf-8')
            lst = line.split('\t')
            par = (int(lst[0]),     # def: User ID,     type: int
                   lst[1])          # def: Gender,      type: str
            self.profile.append(par)
        colum_tag = ['id','gender']
        self.profile = pandas.DataFrame(self.profile, columns=colum_tag)
        print '<profile data loaded, total record ' + str(len(self.profile.index)) + '>'
        # Load social tie data from social.txt
        with open('data/social.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf-8')
            lst = line.split('\t')
            par = (int(lst[0]),     # def: User ID,     type: int
                   int(lst[1]))     # def: Followed ID, type: int
            self.social.append(par)
        colum_tag = ['id','follow_id']
        self.social = pandas.DataFrame(self.social, columns=colum_tag)
        print '<social data loaded, total record ' + str(len(self.social.index)) + '>'
        # Load social tie data from tags.txt
        with open('data/tags.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf-8')
            lst = line.split('\t')
            par = (int(lst[0]),     # def: User ID,     type: int
                   int(lst[1]))     # def: Tag,         type: int
            self.tags.append(par)
        colum_tag = ['id','tag']
        self.tags = pandas.DataFrame(self.tags, columns=colum_tag)
        print '<tags data loaded, total record ' + str(len(self.tags.index)) + '>'
        # Load users' tweets data from tweets.txt
        with open('data/tweets.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf-8')
            lst = line.split('\t')
            par = (int(lst[1]),     # def: User ID,     type: int
                   lst[0])          # def: Tweet,       type: str
            self.tweets.append(par)
        colum_tag = ['id','tweet']
        self.tweets = pandas.DataFrame(self.tweets, columns=colum_tag)
        print '<tweets data loaded, total record ' + str(len(self.tweets.index)) + '>'

    def filter(self, poi_threshold = 10, tag_threshold = 10):
        gpi = self.checkins.groupby('id')
        cnt = gpi.count()
        cnt.columns = [i+'_' for i in cnt.columns]
        cnt['id'] = list(cnt.index)
        cnt = cnt.loc[:,['id','poi_']]

        col = self.checkins.columns
        tmp = pandas.merge(self.checkins, cnt)
        self.checkins = tmp[tmp.poi_ >= poi_threshold].loc[:,col]

        col = self.profile.columns
        tmp = pandas.merge(self.profile, cnt)
        self.profile = tmp[tmp.poi_ >= poi_threshold].loc[:,col]

        col = self.tags.columns
        tmp = pandas.merge(self.tags, cnt)
        self.tags = tmp[tmp.poi_ >= poi_threshold].loc[:,col]

        col = self.social.columns
        tmp = pandas.merge(self.social, cnt)
        self.social = tmp[tmp.poi_ >= poi_threshold].loc[:,col]

        col = self.tweets.columns
        tmp = pandas.merge(self.tweets, cnt)
        self.tweets = tmp[tmp.poi_ >= poi_threshold].loc[:,col]

        gpt = self.tags.groupby('tag')
        cnt = gpt.count()
        cnt.columns = ['id_']
        cnt['tag'] = list(cnt.index)

        col = self.tags.columns
        tmp = pandas.merge(self.tags, cnt)
        self.tags = tmp[tmp.id_ >= tag_threshold].loc[:,col]
