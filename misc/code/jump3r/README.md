# jump3r

[![Build Status](https://travis-ci.org/Sciss/jump3r.svg?branch=master)](https://travis-ci.org/Sciss/jump3r)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/de.sciss/jump3r/badge.svg)](https://maven-badges.herokuapp.com/maven-central/de.sciss/jump3r)

## Statement 

A copy of an unofficial LAME mp3 library port to Java from https://sourceforge.net/projects/jump3r/

I (Hanns Holger Rutz) made the following changes to the project:

- using sbt for building
- moving from LGPL v2+ to LGPL v2.1+
- removed UI (Swing) module
- remove JMA API dependency
- give it a proper namespace - packages are in `de.sciss.jump3r`
- main class is now `de.sciss.jump3r.Main`
- no progress bar boundaries are printed when using `--quiet`
  (matching behaviour of most recent LAME)

For simplicity, the `sbt` shell script [by Paul Phillips](https://github.com/paulp/sbt-extras) is included, 
which is released under BSD 3-clause License. So you use `./sbt` and do not need to install sbt.

- to compile: `sbt compile`
- to create self-contained jar: `sbt assembly`

The Java port is based on LAME 3.98.4. I compared decoding against latest C LAME 3.99.5. They seem identical except
for a delay difference, with jump3r producing slightly shorter output (576 sample frames with the example file;
message says this is to compensate codec delay).

This project is published as a Maven artifact to Maven Central (click badge above). To reference from other sbt build:

    "de.sciss" % "jump3r" % "1.0.4"

Below are the original README and USAGE (adapted for markdown).

## README

jump3r - Java Unofficial MP3 EncodeR

...is a Java version of lame-3.98.4

Java port created by Ken Händel.

Original sources by the authors of Lame:

http://www.sourceforge.net/projects/lame

## USAGE

    java -jar jump3r.jar [options] inputfile [outputfile]

For more options, just type:

    java -jar jump3r.jar --help

### Constant Bitrate Examples:

fixed bit rate jstereo 128 kbps encoding:

    java -jar jump3r.jar sample.wav sample.mp3      

fixed bit rate jstereo 128 kbps encoding, higher quality:  (recommended)

    java -jar jump3r.jar -h sample.wav sample.mp3      

Fast encode, low quality  (no noise shaping)

    java -jar jump3r.jar -f sample.wav sample.mp3     

### Variable Bitrate Examples:

LAME has two types of variable bitrate: ABR and VBR.

ABR is the type of variable bitrate encoding usually found in other
MP3 encoders, Vorbis and AAC. The number of bits is determined by
some metric (like perceptual entropy, or just the number of bits
needed for a certain set of encoding tables), and it is not based on
computing the actual encoding/quantization error. ABR should always
give results equal or better than CBR:

ABR:   (`--abr <x>` means encode with an average bitrate of around `x` kbps)

    java -jar jump3r.jar -h --abr 128  sample.wav sample.mp3

VBR is a true variable bitrate mode which bases the number of bits for
each frame on the measured quantization error relative to the
estimated allowed masking. There are 10 compression levels defined, 
ranging from 0=lowest compression to 9 highest compression. The resulting
filesizes depend on the input material. On typical music you can expect
`-V 5` resulting in files averaging 132 kbps, `-V 2` averaging 200 kbps.

Variable Bitrate (VBR): (use `-V n` to adjust quality/filesize)

    java -jar jump3r.jar -V 2 sample.wav sample.mp3

### LOW BITRATES

At lower bitrates, (like 24 kbps per channel), it is recommended that
you use a 16 kHz sampling rate combined with lowpass filtering.  LAME,
as well as commercial encoders (FhG, Xing) will do this automatically.
However, if you feel there is too much (or not enough) lowpass
filtering, you may need to try different values of the lowpass cutoff
and passband width (`--resample`, `--lowpass` and `--lowpass-width` options).

### options guide:

These options are explained in detail below.

Quality related:

- `-m m/s/j/f/a` : mode selection
- `-q n` : Internal algorithm quality setting 0..9. 
   - 0 = slowest algorithms, but potentially highest quality
   - 9 = faster algorithms, very poor quality
- `-h` : same as `-q2`
- `-f` : same as `-q7`

Constant Bit Rate (CBR)

- `-b n` : set bitrate (8, 16, 24, ..., 320)
- `--freeformat` : produce a free format bitstream.  User must also specify a bitrate with `-b`, between 8 and 640 kbps.

Variable Bit Rate (VBR)

- `-v` : VBR
- `--vbr-old` : use old variable bitrate (VBR) routine
- `--vbr-new` : use new variable bitrate (VBR) routine (default)
- `-V n`: VBR quality setting  (0=highest quality, 9=lowest)
- `-b n` : specify a minimum allowed bitrate (8,16,24,...,320)
- `-B n` : specify a maximum allowed bitrate (8,16,24,...,320)
- `-F` : strictly enforce minimum bitrate
- `-t` : disable VBR informational tag 
- `--nohist`: disable display of VBR bitrate histogram
- `--abr n` : specify average bitrate desired

Operational:

- `-r` : assume input file is raw PCM
- `-s n` : input sampling frequency in kHz (for raw PCM input files)
- `--resample n`: output sampling frequency
- `--mp3input` : input file is an MP3 file.  decode using mpglib/mpg123
- `-x` : swap bytes of input file
- `--scale <arg>` : multiply PCM input by <arg>
- `--scale-l <arg>` : scale channel 0 (left) input (multiply PCM data) by <arg>
- `--scale-r <arg>` : scale channel 1 (right) input (multiply PCM data) by <arg>
- `-a` : downmix stereo input file to mono .mp3
- `-e n/5/c` : de-emphasis
- `-p` : add CRC error protection
- `-c` : mark the encoded file as copyrighted
- `-o` : mark the encoded file as a copy
- `-S` : don't print progress report, VBR histogram
- `--strictly-enforce-ISO` : comply as much as possible to ISO MPEG spec
- `--replaygain-fast` : compute RG fast but slightly inaccurately (default)
- `--replaygain-accurate` : compute RG more accurately and find the peak sample
- `--noreplaygain` : disable ReplayGain analysis
- `--clipdetect` : enable `--replaygain-accurate` and print a message whether clipping occurs and how far the waveform is from full scale
- `--decode` : assume input file is an mp3 file, and decode to wav.
- `-t` : disable writing of WAV header when using `--decode` (decode to raw pcm, native endian format (use `-x` to swap))

ID3 tagging:

- `--tt <title>` : audio/song title (max 30 chars for version 1 tag)
- `--ta <artist>` : audio/song artist (max 30 chars for version 1 tag)
- `--tl <album>` : audio/song album (max 30 chars for version 1 tag)
- `--ty <year>` : audio/song year of issue (1 to 9999)
- `--tc <comment>` : user-defined text (max 30 chars for v1 tag, 28 for v1.1)
- `--tn <track>` : audio/song track number (1 to 255, creates v1.1 tag)
- `--tg <genre>`: audio/song genre (name or number in list)
- `--add-id3v2` : force addition of version 2 tag
- `--id3v1-only` : add only a version 1 tag
- `--id3v2-only` : add only a version 2 tag
- `--space-id3v1` : pad version 1 tag with spaces instead of nulls
- `--pad-id3v2` : same as `--pad-id3v2-size 128`
- `--pad-id3v2-size <num>` : adds version 2 tag, pad with extra `<num>` bytes
- `--genre-list` : print alphabetically sorted ID3 genre list and exit

Note: A version 2 tag will NOT be added unless one of the input fields
won't fit in a version 1 tag (e.g. the title string is longer than 30
characters), or the `--add-id3v2` or `--id3v2-only` options are used.

Windows and OS/2-specific options:

- `--priority <type>` sets the process priority

options not yet described:

- `--nores` : disable bit reservoir
- `--disptime`
- `--lowpass`
- `--lowpass-width`
- `--highpass`
- `--highpass-width`

### Detailed description of all options in alphabetical order

#### downmix

    -a  

mix the stereo input file to mono and encode as mono.  

This option is only needed in the case of raw PCM stereo input 
(because LAME cannot determine the number of channels in the input file).
To encode a stereo PCM input file as mono, use `java -jar jump3r.jar -m s -a`

For WAV and AIFF input files, using `-m m` will always produce a
mono .mp3 file from both mono and stereo input.

#### average bitrate encoding (aka Safe VBR)

    --abr n

turns on encoding with a targeted average bitrate of n kbps, allowing
to use frames of different sizes.  The allowed range of n is 8...320 
kbps, you can use any integer value within that range.

#### bitrate

    -b  n

For MPEG-1 (sampling frequencies of 32, 44.1 and 48 kHz)<br>
n =   32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320

For MPEG-2 (sampling frequencies of 16, 22.05 and 24 kHz)<br>
n = 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160

For MPEG-2.5 (sampling frequencies of 8, 11.025 and 12 kHz)<br>
n = 8, 16, 24, 32, 40, 48, 56, 64


The bitrate to be used.  Default is 128 kbps MPEG1, 80 kbps MPEG2.

When used with variable bitrate encodings (VBR), `-b` specifies the
minimum bitrate to use.  This is useful only if you need to circumvent
a buggy hardware device with strange bitrate constrains.

#### max bitrate

    -B  n

see also option `-b` for allowed bitrates.

Maximum allowed bitrate when using VBR/ABR.

Using `-B` is NOT RECOMMENDED.  A 128 kbps CBR bitstream, because of the
bit reservoir, can actually have frames which use as many bits as a
320 kbps frame.  ABR/VBR modes minimize the use of the bit reservoir, and
thus need to allow 320 kbps frames to get the same flexibility as CBR
streams.  This is useful only if you need to circumvent a buggy hardware
device with strange bitrate constrains.

#### copyright

    -c   

mark the encoded file as copyrighted

#### clipping detection

    --clipdetect

Enable `--replaygain-accurate` and print a message whether clipping 
occurs and how far in dB the waveform is from full scale.

See also: `--replaygain-accurate`

#### mpglib decode capability

    --decode 

This just uses LAME's mpg123/mpglib interface to decode an MP3 file to
a wav file.  The input file can be any input type supported by
encoding, including .mp3 (layers 1, 2 and 3).  

If `-t` is used (disable wav header), LAME will output
raw pcm in native endian format (use `-x` to swap bytes).

#### de-emphasis

    -e  n/5/c   

    n = (none, default)
    5 = 0/15 microseconds
    c = citt j.17

All this does is set a flag in the bitstream.  If you have a PCM
input file where one of the above types of (obsolete) emphasis has
been applied, you can set this flag in LAME.  Then the mp3 decoder
should de-emphasize the output during playback, although most 
decoders ignore this flag.

A better solution would be to apply the de-emphasis with a standalone
utility before encoding, and then encode without `-e`.  

#### fast mode

    -f   

Same as `-q 7`.

NOT RECOMMENDED. Use when encoding speed is critical and encoding
quality does not matter. Disable noise shaping. Psycho acoustics are
used only for bit allocation and pre-echo detection.

#### strictly enforce VBR minimum bitrate

    -F   

strictly enforce VBR minimum bitrate. With out this option, the minimum
bitrate will be ignored for passages of analog silence.

#### free format bitstreams

    --freeformat   

LAME will produce a fixed bitrate, free format bitstream.
User must specify the desired bitrate in kbps, which can
be any integer between 8 and 640.

Not supported by most decoders. Compliant decoders (of which there
are few) are only required to support up to 320 kbps.

Decoders which can handle free format:

                                         supports up to
    MAD                      				640 kbps
    jump3r									550 kbps  
    Freeamp:                 				440 kbps
    l3dec:                   				310 kbps

#### high quality

    -h

use some quality improvements. The same as `-q 2`.

#### Modes

    -m m           mono
    -m s           stereo
    -m j           joint stereo
    -m f           forced mid/side stereo
    -m d           dual (independent) channels
    -m a           auto

MONO is the default mode for mono input files.  If `-m m` is specified
for a stereo input file, the two channels will be averaged into a mono
signal.  

__STEREO__

__JOINT STEREO__ is the default mode for stereo files with fixed bitrates of
128 kbps or less.  At higher fixed bitrates, the default is stereo.
For VBR encoding, jstereo is the default for VBR_q >4, and stereo
is the default for VBR_q <=4.  You can override all of these defaults
by specifying the mode on the command line.  

jstereo means the encoder can use (on a frame by frame bases) either
regular stereo (just encode left and right channels independently)
or mid/side stereo.  In mid/side stereo, the mid (L+R) and side (L-R)
channels are encoded, and more bits are allocated to the mid channel
than the side channel.  This will effectively increase the bandwidth
if the signal does not have too much stereo separation.  

Mid/side stereo is basically a trick to increase bandwidth.  At 128 kbps,
it is clearly worth while.  At higher bitrates it is less useful.

For truly mono content, use `-m m`, which will automatically down
sample your input file to mono.  This will produce 30% better results
over `-m j`.  

Using mid/side stereo inappropriately can result in audible
compression artifacts.  To much switching between mid/side and regular
stereo can also sound bad.  To determine when to switch to mid/side
stereo, LAME uses a much more sophisticated algorithm than that
described in the ISO documentation.

__FORCED MID/SIDE STEREO__ forces all frames to be encoded mid/side stereo.  It 
should only be used if you are sure every frame of the input file
has very little stereo seperation.  

__DUAL CHANNELS__   Not supported.

__AUTO__

Auto select should select (if input is stereo)

          8 kbps   Mono
     16- 96 kbps   Intensity Stereo (if available, otherwise Joint Stereo)
    112-128 kbps   Joint Stereo -mj
    160-192 kbps   -mj with variable mid/side threshold
    224-320 kbps   Independent Stereo -ms

#### MP3 input file

    --mp3input

Assume the input file is a MP3 file.  LAME will decode the input file
before re-encoding it.  Since MP3 is a lossy format, this is 
not recommended in general.  But it is useful for creating low bitrate
mp3s from high bitrate mp3s.  If the filename ends in `.mp3` LAME will assume
it is an MP3.

#### disable histogram display

    --nohist

By default, LAME will display a bitrate histogram while producing
VBR mp3 files.  This will disable that feature.

#### disable ReplayGain analysis

    --noreplaygain

By default ReplayGain analysis is enabled. This switch disables it.

See also: `--replaygain-accurate`, `--replaygain-fast`

#### non-original

    -o   

mark the encoded file as a copy

#### CRC error protection

    -p  

turn on CRC error protection.  
Yes this really does work correctly in LAME.  However, it takes 
16 bits per frame that would otherwise be used for encoding.

#### algorithm quality selection

    -q n  

Bitrate is of course the main influence on quality.  The higher the
bitrate, the higher the quality.  But for a given bitrate,
we have a choice of algorithms to determine the best
scale factors and huffman encoding (noise shaping).

- `-q 0` :  use slowest and best possible version of all algorithms.
- `-q 2` :  recommended.  Same as `-h`.  `-q 0` and `-q 1` are slow and may not produce significantly higher quality.  
- `-q 5` :  default value.  Good speed, reasonable quality
- `-q 7` :  same as `-f`.  Very fast, ok quality.  (psycho acoustics are used for pre-echo and M/S, but no noise shaping is done.  
- `-q 9` :  disables almost all algorithms including psy-model.  poor quality.

#### input file is raw pcm

    -r  

Assume the input file is raw pcm.  Sampling rate and mono/stereo/jstereo
must be specified on the command line.  Without `-r`, LAME will perform
several `fseek()`'s on the input file looking for WAV and AIFF headers.

#### slightly more accurate ReplayGain analysis and finding the peak sample

    --replaygain-accurate

Enable decoding on the fly. Compute "Radio" ReplayGain on the decoded 
data stream. Find the peak sample of the decoded data stream and store 
it in the file.

ReplayGain analysis does _not_ affect the content of a compressed data
stream itself, it is a value stored in the header of a sound file. 
Information on the purpose of ReplayGain and the algorithms used is 
available from http://www.replaygain.org/

By default, LAME performs ReplayGain analysis on the input data (after
the user-specified volume scaling). This behaviour might give slightly 
inaccurate results because the data on the output of a lossy 
compression/decompression sequence differs from the initial input data. 
When `--replaygain-accurate` is specified the mp3 stream gets decoded on
the fly and the analysis is performed on the decoded data stream. 
Although theoretically this method gives more accurate results, it has
several disadvantages:

- tests have shown that the difference between the ReplayGain values 
  computed on the input data and decoded data is usually no greater 
  than 0.5dB, although the minimum volume difference the human ear 
  can perceive is about 1.0dB
- decoding on the fly significantly slows down the encoding process

The apparent advantage is that:

- with `--replaygain-accurate` the peak sample is determined and 
  stored in the file. The knowledge of the peak sample can be useful
  to decoders (players) to prevent a negative effect called 'clipping'
  that introduces distortion into sound.
    
Only the "Radio" ReplayGain value is computed. It is stored in the LAME tag. 
The analysis is performed with the reference volume equal to 89dB. 
Note: the reference volume has been changed from 83dB on transition
from version 3.95 to 3.95.1.

See also: `--replaygain-fast`, `--noreplaygain`, `--clipdetect`

#### fast ReplayGain analysis

    --replaygain-fast

Compute "Radio" ReplayGain of the input data stream after user-specified
volume scaling and/or resampling.

ReplayGain analysis does _not_ affect the content of a compressed data
stream itself, it is a value stored in the header of a sound file. 
Information on the purpose of ReplayGain and the algorithms used is 
available from http://www.replaygain.org/

Only the "Radio" ReplayGain value is computed. It is stored in the LAME tag. 
The analysis is performed with the reference volume equal to 89dB. 
Note: the reference volume has been changed from 83dB on transition
from version 3.95 to 3.95.1.

This switch is enabled by default.

See also: `--replaygain-accurate`, `--noreplaygain`

#### output sampling frequency in kHz

    --resample  n

where n = 8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48

Output sampling frequency.  Resample the input if necessary.  

If not specified, LAME may sometimes resample automatically 
when faced with extreme compression conditions (like encoding
a 44.1 kHz input file at 32 kbps).  To disable this automatic
resampling, you have to use `--resamle` to set the output sample-rate
equal to the input sample-rate.  In that case, LAME will not
perform any extra computations.

#### sampling frequency in kHz

    -s  n

where n = sampling rate in kHz.

Required for raw PCM input files.  Otherwise it will be determined
from the header information in the input file.

LAME will automatically resample the input file to one of the
supported MP3 sample-rates if necessary.

#### silent operation

    -S

don't print progress report

#### scale

    --scale <arg>

Scales input by `<arg>`.  This just multiplies the PCM data
(after it has been converted to floating point) by `<arg>`.  

    <arg> > 1:  increase volume
    <arg> = 1:  no effect
    <arg> < 1:  reduce volume

Use with care, since most MP3 decoders will truncate data
which decodes to values greater than 32768.  

#### strict ISO complience

    --strictly-enforce-ISO   

With this option, LAME will enforce the 7680 bit limitation on
total frame size.  This results in many wasted bits for
high bitrate encodings.

#### disable VBR tag

    -t              

Disable writing of the VBR Tag (only valid if `-v` flag is
specified) This tag in embedded in frame 0 of the MP3 file.  It lets
VBR aware players correctly seek and compute playing times of VBR
files.

When `--decode` is specified (decode mp3 to wav), this flag will 
disable writing the WAV header.  The output will be raw pcm,
native endian format.  Use `-x` to swap bytes.

#### variable bit rate  (VBR)

    -v

Turn on VBR.  There are several ways you can use VBR.  I personally
like using VBR to get files slightly bigger than 128 kbps files, where
the extra bits are used for the occasional difficult-to-encode frame.
For this, try specifying a minimum bitrate to use with VBR:

    java -jar jump3r.jar -v      -b 112  input.wav output.mp3

If the file is too big, use `-V n`, where n = 0...9

    java -jar jump3r.jar -v -V n -b 112  input.wav output.mp3

If you want to use VBR to get the maximum compression possible,
and for this, you can try:  

    java -jar jump3r.jar -v  input.wav output.mp3
    java -jar jump3r.jar -v -V n input.wav output.mp3         (to vary quality/filesize)

#### VBR quality setting

    -V n       

n = 0...9.  Specifies the value of VBR_q.
default = 4,  highest quality = 0, smallest files = 9

Using `-V 6` or higher (lower quality) is NOT RECOMMENDED.  
ABR will produce better results.  

__How is VBR_q used?__

The value of VBR_q influences two basic parameters of LAME's psycho
acoustics:

- the absolute threshold of hearing
- the sample to noise ratio

The lower the VBR_q value the lower the injected quantization noise
will be.
 
*NOTE* No psy-model is perfect, so there can often be distortion which
is audible even though the psy-model claims it is not!  Thus using a
small minimum bitrate can result in some aggressive compression and
audible distortion even with `-V 0`.  Thus using `-V 0` does not sound
better than a fixed 256 kbps encoding.  For example: suppose in the 1 kHz
frequency band the psy-model claims 20 dB of distortion will not be
detectable by the human ear, so LAME VBR-0 will compress that
frequency band as much as possible and introduce at most 20 dB of
distortion.  Using a fixed 256 kbps framesize, LAME could end up
introducing only 2 dB of distortion.  If the psy-model was correct,
they will both sound the same.  If the psy-model was wrong, the VBR-0
result can sound worse.

#### swapbytes   

    -x

swap bytes in the input file (and output file when using `--decode`).
For sorting out little endian/big endian type problems.  If your
encodings sound like static, try this first.
