import utils.clients
import utils.http_session
import utils.transport
import utils.session
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
subject = 1
runs = [4, 8, 12]  # imagined left/right hand movement runs

raw_fnames = eegbci.load_data(subject, runs, path='data')
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

print(raw.info)
print(f"Channels: {raw.ch_names}")
print(f"Duration: {raw.times[-1]:.1f} seconds")

# Filter to mu/beta rhythms (8-30 Hz) — the bands that shift during motor imagery
raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')

# Fix channel naming and set electrode positions for topography plots
mne.datasets.eegbci.standardize(raw)
raw.set_montage('standard_1020', on_missing='warn')

# Epoch around left/right hand imagery events
events, event_id = mne.events_from_annotations(raw)
print(f"Event IDs found: {event_id}")
print(f"Number of events: {len(events)}")

picks = mne.pick_types(raw.info, eeg=True)

epochs = mne.Epochs(
    raw, events, event_id=event_id,
    tmin=-0.5, tmax=3.0,
    picks=picks, baseline=None, preload=True
)
print(epochs)

# Keep only left (T1) and right (T2), drop rest (T0)
epochs_lr = epochs[['T1', 'T2']]
print(f"Left/Right epochs: {epochs_lr}")

labels = epochs_lr.events[:, -1] - 2  # T1(2)->0, T2(3)->1
X = epochs_lr.get_data(copy=False)

# CSP feature extraction + SVM classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

n_components = 6
csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp = csp.transform(X_test)

clf = SVC(kernel='linear')
clf.fit(X_train_csp, y_train)
y_pred = clf.predict(X_test_csp)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.1f}%")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

# Visualize results
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Left', 'Right'], ax=ax)
ax.set_title(f'Confusion Matrix (Accuracy: {acc*100:.1f}%)')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

csp.plot_patterns(
    epochs_lr.info, ch_type='eeg', units='Patterns (AU)',
    size=3, components=list(range(n_components))
)
plt.savefig('results/scalp_topography.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved results/confusion_matrix.png and results/scalp_topography.png")