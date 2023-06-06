import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
        
training = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training,test])

#print(all_data.columns)
#print(all_data.describe())

# Load relevant data
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

all_data.Age = all_data.Age.fillna(training.Age.median())
all_data.Fare = all_data.Fare.fillna(training.Fare.median())
all_data.dropna(subset=['Embarked'],inplace = True)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()
all_data.Pclass = all_data.Pclass.astype(str)

all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

# Scale data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies[['Age','SibSp','Parch','norm_fare']])
#print(all_dummies)

X_train_scaled = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived


# Naive Bayes
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

cv = cross_val_score(GaussianNB(), X_train_scaled, y_train, cv=5)
#print(cv.mean)

voting_clf = VotingClassifier(estimators = [("NB", GaussianNB())], voting = 'soft') 
voting_clf.fit(X_train_scaled,y_train)

y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)
submission = {'PassengerId': test.PassengerId, 'Survived': y_hat_base_vc}
submission = pd.DataFrame(data=submission)
submission.to_csv('submission.csv', index=False)
