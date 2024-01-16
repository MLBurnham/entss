import entss as en
from cmdstanpy import CmdStanModel

df = en.load_newsletters()
df = df[['text', 'Last Name']]
keywords = ['biden', 'trump']
dimensions = ['supports', 'opposes', 'neutral']

mrclean = en.Cleaner(keyword_list = keywords)

df = mrclean.clean(df, synonyms = False, scrub = True, split = True, keywords = True)

labeler = en.Classifier(targets = keywords, dimensions = dimensions)

df = labeler.label(df, aggregate_on = None)

banks = en.Scaler()

fit, sum = banks.stan_fit(df, targets = keywords, dimensions = dimensions, left_init_cols = 'trump_opposes', right_init_cols = 'trump_supports')

print(fit)
print(sum)