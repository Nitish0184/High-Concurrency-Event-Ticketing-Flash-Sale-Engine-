Skip to content
Nitish0184
Mlflow-Advanced
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Mlflow-Advanced
/
Name your file...
in
main

Edit

Preview
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298
299
300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329
330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439
440
441
442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy # Used to save the best model weights during Early Stopping
import time

def train_mobilenet():
    # ==========================================
    # 1. HARDWARE & GENERAL PARAMETERS
    # ==========================================
    # Automatically target your Samsung-provided GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = 5       # Set to your specific 5 classes
    batch_size = 128      # MobileNet is tiny, so your 16GB VRAM can easily handle 128 images at once
    num_workers = 8       # Use all 8 CPU cores for data loading
    
    # Early Stopping Parameters
    max_epochs = 100      # Maximum number of times to loop through the 22k images
    patience = 15         # If Validation Loss doesn't improve for 15 epochs, stop training

    # ==========================================
    # 2. DATA PREPARATION & TRANSFORMATIONS
    # ==========================================
    # MobileNet expects 224x224 images and standard ImageNet normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),       # Resizes your 440x440 images
            transforms.RandomHorizontalFlip(),   # Basic data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Point this to your main dataset folder containing 'train' and 'val' subfolders
    data_dir = './dataset' 
    
    image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f"Data loaded: {dataset_sizes['train']} Train images, {dataset_sizes['val']} Val images across {num_classes} classes.")

    # ==========================================
    # 3. TRANSFER LEARNING (MODEL SETUP)
    # ==========================================
    # Load the pretrained MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # FREEZE the base layers so we don't destroy the pre-learned edge/shape detection
    for param in model.parameters():
        param.requires_grad = False

    # MobileNet's final layer is stored in 'model.classifier'. 
    # We replace it with a new layer designed specifically for your 5 classes.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Send the modified model to the 16GB GPU
    model = model.to(device)

    # Define the judge (Loss) and the mechanic (Optimizer)
    criterion = nn.CrossEntropyLoss()
    # We only train the brand new classifier layer we just attached
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # ==========================================
    # 4. THE TRAINING LOOP & EARLY STOPPING
    # ==========================================
    since = time.time()

    # Trackers for Early Stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf') # Set initial best loss to infinity
    epochs_no_improve = 0        # Counter for patience

    for epoch in range(max_epochs):
        print(f'\nEpoch {epoch+1}/{max_epochs}')
        print('-' * 15)

        # Each epoch has a training phase and a validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data in batches of 128
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Clear old math from the previous batch

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass (only update weights during the 'train' phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Tally up the loss and correct guesses
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate the final loss and accuracy for this entire epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # --- EARLY STOPPING LOGIC ---
            if phase == 'val':
                # If the validation loss is the lowest we've seen so far, save the model!
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 # Reset patience counter
                    print(f"  [*] Validation Loss improved! Saving model...")
                else:
                    epochs_no_improve += 1
                    print(f"  [!] No improvement in Val Loss for {epochs_no_improve} epoch(s).")

        # Check if we have run out of patience
        if epochs_no_improve >= patience:
            print(f"\n[ALERT] Early stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            break # Breaks out of the main epoch loop

    # ==========================================
    # 5. WRAP UP
    # ==========================================
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    # Load the best weights back into the model before returning it
    model.load_state_dict(best_model_wts)
    
    # Save the final optimized model to your hard drive
    torch.save(model.state_dict(), 'best_mobilenet_baseline.pth')
    print("Saved the best model to 'best_mobilenet_baseline.pth'")

if __name__ == '__main__':
    train_mobilenet()



############################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy
import time

def train_resnet():
    # ==========================================
    # 0. CHOOSE YOUR MODEL HERE
    # ==========================================
    # Change this to 'resnet50' when you want to run the heavier model
    MODEL_CHOICE = 'resnet18' 

    # ==========================================
    # 1. HARDWARE & GENERAL PARAMETERS
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 5       
    num_workers = 8       
    
    # Auto-adjust batch size to protect your 16GB VRAM
    if MODEL_CHOICE == 'resnet18':
        batch_size = 64  # ResNet18 is lighter, 64 is safe
    else:
        batch_size = 32  # ResNet50 is massive, 32 prevents OOM crashes

    # Early Stopping Parameters
    max_epochs = 100      
    patience = 15         

    # ==========================================
    # 2. DATA PREPARATION & TRANSFORMATIONS
    # ==========================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),       # Cleanly resizes your 440x440 images
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Automatically points to your "images/train" and "images/val" folders
    data_dir = './images' 
    
    image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f"Data loaded: {dataset_sizes['train']} Train images, {dataset_sizes['val']} Val images.")
    print(f"Using Model: {MODEL_CHOICE.upper()} | Batch Size: {batch_size}")

    # ==========================================
    # 3. TRANSFER LEARNING (MODEL SETUP)
    # ==========================================
    # Load the correct pre-trained weights based on your choice
    if MODEL_CHOICE == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    # FREEZE the base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final head (Notice we use .fc here instead of .classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # ==========================================
    # 4. THE TRAINING LOOP & EARLY STOPPING
    # ==========================================
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf') 
    epochs_no_improve = 0        

    for epoch in range(max_epochs):
        print(f'\nEpoch {epoch+1}/{max_epochs}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() 

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # EARLY STOPPING LOGIC
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 
                    print(f"  [*] Validation Loss improved! Saving {MODEL_CHOICE} model...")
                else:
                    epochs_no_improve += 1
                    print(f"  [!] No improvement in Val Loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n[ALERT] Early stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            break 

    # ==========================================
    # 5. WRAP UP
    # ==========================================
    time_elapsed = time.time() - since
    print(f'\n{MODEL_CHOICE.upper()} Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    model.load_state_dict(best_model_wts)
    
    # Dynamically save the file name based on the model you ran
    save_path = f'best_{MODEL_CHOICE}_baseline.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved the best model to '{save_path}'")

if __name__ == '__main__':
    train_resnet()


######################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy
import time

def train_vgg16():
    # ==========================================
    # 1. HARDWARE & GENERAL PARAMETERS
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 5       
    num_workers = 8       
    
    # CRITICAL: VGG16 is massive. 32 is usually the maximum safe limit for 16GB VRAM.
    # If you get a "CUDA Out of Memory" error, change this immediately to 16.
    batch_size = 32  

    max_epochs = 100      
    patience = 15         

    # ==========================================
    # 2. DATA PREPARATION & TRANSFORMATIONS
    # ==========================================
    # VGG16 also expects the standard 224x224 input size
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),       
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './images' 
    
    image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f"Data loaded: {dataset_sizes['train']} Train images, {dataset_sizes['val']} Val images.")
    print(f"Using Model: VGG16 | Batch Size: {batch_size}")

    # ==========================================
    # 3. TRANSFER LEARNING (MODEL SETUP)
    # ==========================================
    # Load the heavyweight pre-trained VGG16
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)

    # FREEZE the base layers
    for param in model.parameters():
        param.requires_grad = False

    # VGG16's classifier is a sequence of layers. 
    # We only want to rip out and replace the final one, which is at index [6].
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # We explicitly tell the optimizer to ONLY update that brand new layer at index 6
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

    # ==========================================
    # 4. THE TRAINING LOOP & EARLY STOPPING
    # ==========================================
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf') 
    epochs_no_improve = 0        

    for epoch in range(max_epochs):
        print(f'\nEpoch {epoch+1}/{max_epochs}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() 

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # EARLY STOPPING LOGIC
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 
                    print(f"  [*] Validation Loss improved! Saving VGG16 model...")
                else:
                    epochs_no_improve += 1
                    print(f"  [!] No improvement in Val Loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n[ALERT] Early stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            break 

    # ==========================================
    # 5. WRAP UP
    # ==========================================
    time_elapsed = time.time() - since
    print(f'\nVGG16 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    model.load_state_dict(best_model_wts)
    
    # Save the final optimized VGG16 model
    save_path = 'best_vgg16_baseline.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved the best model to '{save_path}'")

if __name__ == '__main__':
    train_vgg16()
