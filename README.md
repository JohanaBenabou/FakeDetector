# FakeDetector

Problem :
To motivate our project, we are going to imagine a concrete situation. Let's say that Facebook contacted us because they need us for a mission. They want to detect fake profiles without checking all their users which could take infinite times. For that, they noticed that many fake accounts are using pictures from the website thispersondoesnotexist.com and they want a model able to detect if a profile picture is potentially coming from this website or not.

To construct the dataset, we took 10000 fake faces from https://archive.org/details/1mFakeFaces and 10000 genuine faces from https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

Then, we splitted this 20000 images into a train set, a validation set and a test set following the repartition 80/10/10.

The goal of the model is to reduce the volume of suspicious people to check. Facebook will based their future investigations on this model and they count on us ! Let's say that they will ask special pictures to profile which has been detected as potentially fake for instance.

According to an article of the NY times, "Facebook was cautious in removing manually created accounts because it didn’t want to erase authentic profiles" (https://www.nytimes.com/2019/01/30/technology/facebook-fake-accounts.html)

If the model says to Facebook that a profile is suspicious meanwhile it was not the case, then we will consider that Facebook loose time asking more information to confirm the authenticity of the profile. Let's consider that someone is payed to make inquiries about suspicious profiles. The minimum salary in USA is 7.25$/hour (Source : https://www.dol.gov/general/topic/wages/minimumwage#:~:text=The%20federal%20minimum%20wage%20for,of%20the%20two%20minimum%20wages.) . If we consider that it takes 1 day for an employee to judge the authenticity of a profile, make an inquiry about a suspicous profile has a cost of (7.25x24=) 174$. Now, even if a profile proposed by the model is really fake, we have to take into account this investigation cost.

If the model says to Facebook that the profile sounds good meanwhile it was a fake profile, we can consider that the person behind the profile may rip off 10 people that could decide to delete their account after this event. According to this study https://www.quora.com/How-much-money-does-Facebook-make-from-a-single-user-using-the-site-for-1-hour , a user brings 0.05 dollards per hour to Facebook on average. Hence, in terms of annually cost, Facebook is loosing (0.05x24x365)=438$.

Hence, we could derive the following annual matrix cost per member to check :

Predicted \ Truth	0	1
0	0	438
1	174	174
What we want is to design a model that will minimize the cost on our test set.

An other relevant metric for our problem is the F1-score on the positive class (to be fake). Indeed, maximizing this F1-score foster the model to make his best to be precise and efficient on the detection of fake profile. The precision on the positive class is not as important as the recall but being too less precise will results in a big loss since each inquiry has a cost.

At first we are going to test simple models and then we will test more complex ones. We are going also to start with a small resolution. Instead of considering 1024x1024 resolution, we will resize the images into 32x32 and standardize them so that we can work on classical ML models without loosing too many information and too many time during training. For that we propose to use Pytorch that has really great tools to help us doing that.
