/*
 *      MP3 window subband -> subband filtering -> mdct routine
 *
 *      Copyright (c) 1999-2000 Takehiro Tominaga
 *
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
/*
 *         Special Thanks to Patrick De Smet for your advices.
 */

/* $Id: NewMDCT.java,v 1.11 2011/05/24 20:48:06 kenchis Exp $ */

package de.sciss.jump3r.mp3;

import java.util.Arrays;

public class NewMDCT {

	private static final float enwindow[] = {
			-4.77e-07f * 0.740951125354959f / 2.384e-06f,
			1.03951e-04f * 0.740951125354959f / 2.384e-06f,
			9.53674e-04f * 0.740951125354959f / 2.384e-06f,
			2.841473e-03f * 0.740951125354959f / 2.384e-06f,
			3.5758972e-02f * 0.740951125354959f / 2.384e-06f,
			3.401756e-03f * 0.740951125354959f / 2.384e-06f,
			9.83715e-04f * 0.740951125354959f / 2.384e-06f,
			9.9182e-05f * 0.740951125354959f / 2.384e-06f, /* 15 */
			1.2398e-05f * 0.740951125354959f / 2.384e-06f,
			1.91212e-04f * 0.740951125354959f / 2.384e-06f,
			2.283096e-03f * 0.740951125354959f / 2.384e-06f,
			1.6994476e-02f * 0.740951125354959f / 2.384e-06f,
			-1.8756866e-02f * 0.740951125354959f / 2.384e-06f,
			-2.630711e-03f * 0.740951125354959f / 2.384e-06f,
			-2.47478e-04f * 0.740951125354959f / 2.384e-06f,
			-1.4782e-05f * 0.740951125354959f / 2.384e-06f,
			9.063471690191471e-01f, 1.960342806591213e-01f,

			-4.77e-07f * 0.773010453362737f / 2.384e-06f,
			1.05858e-04f * 0.773010453362737f / 2.384e-06f,
			9.30786e-04f * 0.773010453362737f / 2.384e-06f,
			2.521515e-03f * 0.773010453362737f / 2.384e-06f,
			3.5694122e-02f * 0.773010453362737f / 2.384e-06f,
			3.643036e-03f * 0.773010453362737f / 2.384e-06f,
			9.91821e-04f * 0.773010453362737f / 2.384e-06f,
			9.6321e-05f * 0.773010453362737f / 2.384e-06f, /* 14 */
			1.1444e-05f * 0.773010453362737f / 2.384e-06f,
			1.65462e-04f * 0.773010453362737f / 2.384e-06f,
			2.110004e-03f * 0.773010453362737f / 2.384e-06f,
			1.6112804e-02f * 0.773010453362737f / 2.384e-06f,
			-1.9634247e-02f * 0.773010453362737f / 2.384e-06f,
			-2.803326e-03f * 0.773010453362737f / 2.384e-06f,
			-2.77042e-04f * 0.773010453362737f / 2.384e-06f,
			-1.6689e-05f * 0.773010453362737f / 2.384e-06f,
			8.206787908286602e-01f, 3.901806440322567e-01f,

			-4.77e-07f * 0.803207531480645f / 2.384e-06f,
			1.07288e-04f * 0.803207531480645f / 2.384e-06f,
			9.02653e-04f * 0.803207531480645f / 2.384e-06f,
			2.174854e-03f * 0.803207531480645f / 2.384e-06f,
			3.5586357e-02f * 0.803207531480645f / 2.384e-06f,
			3.858566e-03f * 0.803207531480645f / 2.384e-06f,
			9.95159e-04f * 0.803207531480645f / 2.384e-06f,
			9.3460e-05f * 0.803207531480645f / 2.384e-06f, /* 13 */
			1.0014e-05f * 0.803207531480645f / 2.384e-06f,
			1.40190e-04f * 0.803207531480645f / 2.384e-06f,
			1.937389e-03f * 0.803207531480645f / 2.384e-06f,
			1.5233517e-02f * 0.803207531480645f / 2.384e-06f,
			-2.0506859e-02f * 0.803207531480645f / 2.384e-06f,
			-2.974033e-03f * 0.803207531480645f / 2.384e-06f,
			-3.07560e-04f * 0.803207531480645f / 2.384e-06f,
			-1.8120e-05f * 0.803207531480645f / 2.384e-06f,
			7.416505462720353e-01f, 5.805693545089249e-01f,

			-4.77e-07f * 0.831469612302545f / 2.384e-06f,
			1.08242e-04f * 0.831469612302545f / 2.384e-06f,
			8.68797e-04f * 0.831469612302545f / 2.384e-06f,
			1.800537e-03f * 0.831469612302545f / 2.384e-06f,
			3.5435200e-02f * 0.831469612302545f / 2.384e-06f,
			4.049301e-03f * 0.831469612302545f / 2.384e-06f,
			9.94205e-04f * 0.831469612302545f / 2.384e-06f,
			9.0599e-05f * 0.831469612302545f / 2.384e-06f, /* 12 */
			9.060e-06f * 0.831469612302545f / 2.384e-06f,
			1.16348e-04f * 0.831469612302545f / 2.384e-06f,
			1.766682e-03f * 0.831469612302545f / 2.384e-06f,
			1.4358521e-02f * 0.831469612302545f / 2.384e-06f,
			-2.1372318e-02f * 0.831469612302545f / 2.384e-06f,
			-3.14188e-03f * 0.831469612302545f / 2.384e-06f,
			-3.39031e-04f * 0.831469612302545f / 2.384e-06f,
			-1.9550e-05f * 0.831469612302545f / 2.384e-06f,
			6.681786379192989e-01f, 7.653668647301797e-01f,

			-4.77e-07f * 0.857728610000272f / 2.384e-06f,
			1.08719e-04f * 0.857728610000272f / 2.384e-06f,
			8.29220e-04f * 0.857728610000272f / 2.384e-06f,
			1.399517e-03f * 0.857728610000272f / 2.384e-06f,
			3.5242081e-02f * 0.857728610000272f / 2.384e-06f,
			4.215240e-03f * 0.857728610000272f / 2.384e-06f,
			9.89437e-04f * 0.857728610000272f / 2.384e-06f,
			8.7261e-05f * 0.857728610000272f / 2.384e-06f, /* 11 */
			8.106e-06f * 0.857728610000272f / 2.384e-06f,
			9.3937e-05f * 0.857728610000272f / 2.384e-06f,
			1.597881e-03f * 0.857728610000272f / 2.384e-06f,
			1.3489246e-02f * 0.857728610000272f / 2.384e-06f,
			-2.2228718e-02f * 0.857728610000272f / 2.384e-06f,
			-3.306866e-03f * 0.857728610000272f / 2.384e-06f,
			-3.71456e-04f * 0.857728610000272f / 2.384e-06f,
			-2.1458e-05f * 0.857728610000272f / 2.384e-06f,
			5.993769336819237e-01f, 9.427934736519954e-01f,

			-4.77e-07f * 0.881921264348355f / 2.384e-06f,
			1.08719e-04f * 0.881921264348355f / 2.384e-06f,
			7.8392e-04f * 0.881921264348355f / 2.384e-06f,
			9.71317e-04f * 0.881921264348355f / 2.384e-06f,
			3.5007000e-02f * 0.881921264348355f / 2.384e-06f,
			4.357815e-03f * 0.881921264348355f / 2.384e-06f,
			9.80854e-04f * 0.881921264348355f / 2.384e-06f,
			8.3923e-05f * 0.881921264348355f / 2.384e-06f, /* 10 */
			7.629e-06f * 0.881921264348355f / 2.384e-06f,
			7.2956e-05f * 0.881921264348355f / 2.384e-06f,
			1.432419e-03f * 0.881921264348355f / 2.384e-06f,
			1.2627602e-02f * 0.881921264348355f / 2.384e-06f,
			-2.3074150e-02f * 0.881921264348355f / 2.384e-06f,
			-3.467083e-03f * 0.881921264348355f / 2.384e-06f,
			-4.04358e-04f * 0.881921264348355f / 2.384e-06f,
			-2.3365e-05f * 0.881921264348355f / 2.384e-06f,
			5.345111359507916e-01f, 1.111140466039205e+00f,

			-9.54e-07f * 0.903989293123443f / 2.384e-06f,
			1.08242e-04f * 0.903989293123443f / 2.384e-06f,
			7.31945e-04f * 0.903989293123443f / 2.384e-06f,
			5.15938e-04f * 0.903989293123443f / 2.384e-06f,
			3.4730434e-02f * 0.903989293123443f / 2.384e-06f,
			4.477024e-03f * 0.903989293123443f / 2.384e-06f,
			9.68933e-04f * 0.903989293123443f / 2.384e-06f,
			8.0585e-05f * 0.903989293123443f / 2.384e-06f, /* 9 */
			6.676e-06f * 0.903989293123443f / 2.384e-06f,
			5.2929e-05f * 0.903989293123443f / 2.384e-06f,
			1.269817e-03f * 0.903989293123443f / 2.384e-06f,
			1.1775017e-02f * 0.903989293123443f / 2.384e-06f,
			-2.3907185e-02f * 0.903989293123443f / 2.384e-06f,
			-3.622532e-03f * 0.903989293123443f / 2.384e-06f,
			-4.38213e-04f * 0.903989293123443f / 2.384e-06f,
			-2.5272e-05f * 0.903989293123443f / 2.384e-06f,
			4.729647758913199e-01f, 1.268786568327291e+00f,

			-9.54e-07f * 0.92387953251128675613f / 2.384e-06f,
			1.06812e-04f * 0.92387953251128675613f / 2.384e-06f,
			6.74248e-04f * 0.92387953251128675613f / 2.384e-06f,
			3.3379e-05f * 0.92387953251128675613f / 2.384e-06f,
			3.4412861e-02f * 0.92387953251128675613f / 2.384e-06f,
			4.573822e-03f * 0.92387953251128675613f / 2.384e-06f,
			9.54151e-04f * 0.92387953251128675613f / 2.384e-06f,
			7.6771e-05f * 0.92387953251128675613f / 2.384e-06f,
			6.199e-06f * 0.92387953251128675613f / 2.384e-06f,
			3.4332e-05f * 0.92387953251128675613f / 2.384e-06f,
			1.111031e-03f * 0.92387953251128675613f / 2.384e-06f,
			1.0933399e-02f * 0.92387953251128675613f / 2.384e-06f,
			-2.4725437e-02f * 0.92387953251128675613f / 2.384e-06f,
			-3.771782e-03f * 0.92387953251128675613f / 2.384e-06f,
			-4.72546e-04f * 0.92387953251128675613f / 2.384e-06f,
			-2.7657e-05f * 0.92387953251128675613f / 2.384e-06f,
			4.1421356237309504879e-01f, /* tan(PI/8) */
			1.414213562373095e+00f,

			-9.54e-07f * 0.941544065183021f / 2.384e-06f,
			1.05381e-04f * 0.941544065183021f / 2.384e-06f,
			6.10352e-04f * 0.941544065183021f / 2.384e-06f,
			-4.75883e-04f * 0.941544065183021f / 2.384e-06f,
			3.4055710e-02f * 0.941544065183021f / 2.384e-06f,
			4.649162e-03f * 0.941544065183021f / 2.384e-06f,
			9.35555e-04f * 0.941544065183021f / 2.384e-06f,
			7.3433e-05f * 0.941544065183021f / 2.384e-06f, /* 7 */
			5.245e-06f * 0.941544065183021f / 2.384e-06f,
			1.7166e-05f * 0.941544065183021f / 2.384e-06f,
			9.56535e-04f * 0.941544065183021f / 2.384e-06f,
			1.0103703e-02f * 0.941544065183021f / 2.384e-06f,
			-2.5527000e-02f * 0.941544065183021f / 2.384e-06f,
			-3.914356e-03f * 0.941544065183021f / 2.384e-06f,
			-5.07355e-04f * 0.941544065183021f / 2.384e-06f,
			-3.0041e-05f * 0.941544065183021f / 2.384e-06f,
			3.578057213145241e-01f, 1.546020906725474e+00f,

			-9.54e-07f * 0.956940335732209f / 2.384e-06f,
			1.02520e-04f * 0.956940335732209f / 2.384e-06f,
			5.39303e-04f * 0.956940335732209f / 2.384e-06f,
			-1.011848e-03f * 0.956940335732209f / 2.384e-06f,
			3.3659935e-02f * 0.956940335732209f / 2.384e-06f,
			4.703045e-03f * 0.956940335732209f / 2.384e-06f,
			9.15051e-04f * 0.956940335732209f / 2.384e-06f,
			7.0095e-05f * 0.956940335732209f / 2.384e-06f, /* 6 */
			4.768e-06f * 0.956940335732209f / 2.384e-06f,
			9.54e-07f * 0.956940335732209f / 2.384e-06f,
			8.06808e-04f * 0.956940335732209f / 2.384e-06f,
			9.287834e-03f * 0.956940335732209f / 2.384e-06f,
			-2.6310921e-02f * 0.956940335732209f / 2.384e-06f,
			-4.048824e-03f * 0.956940335732209f / 2.384e-06f,
			-5.42164e-04f * 0.956940335732209f / 2.384e-06f,
			-3.2425e-05f * 0.956940335732209f / 2.384e-06f,
			3.033466836073424e-01f, 1.662939224605090e+00f,

			-1.431e-06f * 0.970031253194544f / 2.384e-06f,
			9.9182e-05f * 0.970031253194544f / 2.384e-06f,
			4.62532e-04f * 0.970031253194544f / 2.384e-06f,
			-1.573563e-03f * 0.970031253194544f / 2.384e-06f,
			3.3225536e-02f * 0.970031253194544f / 2.384e-06f,
			4.737377e-03f * 0.970031253194544f / 2.384e-06f,
			8.91685e-04f * 0.970031253194544f / 2.384e-06f,
			6.6280e-05f * 0.970031253194544f / 2.384e-06f, /* 5 */
			4.292e-06f * 0.970031253194544f / 2.384e-06f,
			-1.3828e-05f * 0.970031253194544f / 2.384e-06f,
			6.61850e-04f * 0.970031253194544f / 2.384e-06f,
			8.487225e-03f * 0.970031253194544f / 2.384e-06f,
			-2.7073860e-02f * 0.970031253194544f / 2.384e-06f,
			-4.174709e-03f * 0.970031253194544f / 2.384e-06f,
			-5.76973e-04f * 0.970031253194544f / 2.384e-06f,
			-3.4809e-05f * 0.970031253194544f / 2.384e-06f,
			2.504869601913055e-01f, 1.763842528696710e+00f,

			-1.431e-06f * 0.98078528040323f / 2.384e-06f,
			9.5367e-05f * 0.98078528040323f / 2.384e-06f,
			3.78609e-04f * 0.98078528040323f / 2.384e-06f,
			-2.161503e-03f * 0.98078528040323f / 2.384e-06f,
			3.2754898e-02f * 0.98078528040323f / 2.384e-06f,
			4.752159e-03f * 0.98078528040323f / 2.384e-06f,
			8.66413e-04f * 0.98078528040323f / 2.384e-06f,
			6.2943e-05f * 0.98078528040323f / 2.384e-06f, /* 4 */
			3.815e-06f * 0.98078528040323f / 2.384e-06f,
			-2.718e-05f * 0.98078528040323f / 2.384e-06f,
			5.22137e-04f * 0.98078528040323f / 2.384e-06f,
			7.703304e-03f * 0.98078528040323f / 2.384e-06f,
			-2.7815342e-02f * 0.98078528040323f / 2.384e-06f,
			-4.290581e-03f * 0.98078528040323f / 2.384e-06f,
			-6.11782e-04f * 0.98078528040323f / 2.384e-06f,
			-3.7670e-05f * 0.98078528040323f / 2.384e-06f,
			1.989123673796580e-01f, 1.847759065022573e+00f,

			-1.907e-06f * 0.989176509964781f / 2.384e-06f,
			9.0122e-05f * 0.989176509964781f / 2.384e-06f,
			2.88486e-04f * 0.989176509964781f / 2.384e-06f,
			-2.774239e-03f * 0.989176509964781f / 2.384e-06f,
			3.2248020e-02f * 0.989176509964781f / 2.384e-06f,
			4.748821e-03f * 0.989176509964781f / 2.384e-06f,
			8.38757e-04f * 0.989176509964781f / 2.384e-06f,
			5.9605e-05f * 0.989176509964781f / 2.384e-06f, /* 3 */
			3.338e-06f * 0.989176509964781f / 2.384e-06f,
			-3.9577e-05f * 0.989176509964781f / 2.384e-06f,
			3.88145e-04f * 0.989176509964781f / 2.384e-06f,
			6.937027e-03f * 0.989176509964781f / 2.384e-06f,
			-2.8532982e-02f * 0.989176509964781f / 2.384e-06f,
			-4.395962e-03f * 0.989176509964781f / 2.384e-06f,
			-6.46591e-04f * 0.989176509964781f / 2.384e-06f,
			-4.0531e-05f * 0.989176509964781f / 2.384e-06f,
			1.483359875383474e-01f, 1.913880671464418e+00f,

			-1.907e-06f * 0.995184726672197f / 2.384e-06f,
			8.4400e-05f * 0.995184726672197f / 2.384e-06f,
			1.91689e-04f * 0.995184726672197f / 2.384e-06f,
			-3.411293e-03f * 0.995184726672197f / 2.384e-06f,
			3.1706810e-02f * 0.995184726672197f / 2.384e-06f,
			4.728317e-03f * 0.995184726672197f / 2.384e-06f,
			8.09669e-04f * 0.995184726672197f / 2.384e-06f,
			5.579e-05f * 0.995184726672197f / 2.384e-06f,
			3.338e-06f * 0.995184726672197f / 2.384e-06f,
			-5.0545e-05f * 0.995184726672197f / 2.384e-06f,
			2.59876e-04f * 0.995184726672197f / 2.384e-06f,
			6.189346e-03f * 0.995184726672197f / 2.384e-06f,
			-2.9224873e-02f * 0.995184726672197f / 2.384e-06f,
			-4.489899e-03f * 0.995184726672197f / 2.384e-06f,
			-6.80923e-04f * 0.995184726672197f / 2.384e-06f,
			-4.3392e-05f * 0.995184726672197f / 2.384e-06f,
			9.849140335716425e-02f, 1.961570560806461e+00f,

			-2.384e-06f * 0.998795456205172f / 2.384e-06f,
			7.7724e-05f * 0.998795456205172f / 2.384e-06f,
			8.8215e-05f * 0.998795456205172f / 2.384e-06f,
			-4.072189e-03f * 0.998795456205172f / 2.384e-06f,
			3.1132698e-02f * 0.998795456205172f / 2.384e-06f,
			4.691124e-03f * 0.998795456205172f / 2.384e-06f,
			7.79152e-04f * 0.998795456205172f / 2.384e-06f,
			5.2929e-05f * 0.998795456205172f / 2.384e-06f,
			2.861e-06f * 0.998795456205172f / 2.384e-06f,
			-6.0558e-05f * 0.998795456205172f / 2.384e-06f,
			1.37329e-04f * 0.998795456205172f / 2.384e-06f,
			5.462170e-03f * 0.998795456205172f / 2.384e-06f,
			-2.9890060e-02f * 0.998795456205172f / 2.384e-06f,
			-4.570484e-03f * 0.998795456205172f / 2.384e-06f,
			-7.14302e-04f * 0.998795456205172f / 2.384e-06f,
			-4.6253e-05f * 0.998795456205172f / 2.384e-06f,
			4.912684976946725e-02f, 1.990369453344394e+00f,

			3.5780907e-02f * Util.SQRT2 * 0.5f / 2.384e-06f,
			1.7876148e-02f * Util.SQRT2 * 0.5f / 2.384e-06f,
			3.134727e-03f * Util.SQRT2 * 0.5f / 2.384e-06f,
			2.457142e-03f * Util.SQRT2 * 0.5f / 2.384e-06f,
			9.71317e-04f * Util.SQRT2 * 0.5f / 2.384e-06f,
			2.18868e-04f * Util.SQRT2 * 0.5f / 2.384e-06f,
			1.01566e-04f * Util.SQRT2 * 0.5f / 2.384e-06f,
			1.3828e-05f * Util.SQRT2 * 0.5f / 2.384e-06f,

			3.0526638e-02f / 2.384e-06f, 4.638195e-03f / 2.384e-06f,
			7.47204e-04f / 2.384e-06f, 4.9591e-05f / 2.384e-06f,
			4.756451e-03f / 2.384e-06f, 2.1458e-05f / 2.384e-06f,
			-6.9618e-05f / 2.384e-06f, /* 2.384e-06/2.384e-06 */
	};

	private static final int NS = 12;
	private static final int NL = 36;

	private static final float win[][] = {
	    {
	     2.382191739347913e-13f,
	     6.423305872147834e-13f,
	     9.400849094049688e-13f,
	     1.122435026096556e-12f,
	     1.183840321267481e-12f,
	     1.122435026096556e-12f,
	     9.400849094049690e-13f,
	     6.423305872147839e-13f,
	     2.382191739347918e-13f,

	     5.456116108943412e-12f,
	     4.878985199565852e-12f,
	     4.240448995017367e-12f,
	     3.559909094758252e-12f,
	     2.858043359288075e-12f,
	     2.156177623817898e-12f,
	     1.475637723558783e-12f,
	     8.371015190102974e-13f,
	     2.599706096327376e-13f,

	     -5.456116108943412e-12f,
	     -4.878985199565852e-12f,
	     -4.240448995017367e-12f,
	     -3.559909094758252e-12f,
	     -2.858043359288076e-12f,
	     -2.156177623817898e-12f,
	     -1.475637723558783e-12f,
	     -8.371015190102975e-13f,
	     -2.599706096327376e-13f,

	     -2.382191739347923e-13f,
	     -6.423305872147843e-13f,
	     -9.400849094049696e-13f,
	     -1.122435026096556e-12f,
	     -1.183840321267481e-12f,
	     -1.122435026096556e-12f,
	     -9.400849094049694e-13f,
	     -6.423305872147840e-13f,
	     -2.382191739347918e-13f,
	     },
	    {
	     2.382191739347913e-13f,
	     6.423305872147834e-13f,
	     9.400849094049688e-13f,
	     1.122435026096556e-12f,
	     1.183840321267481e-12f,
	     1.122435026096556e-12f,
	     9.400849094049688e-13f,
	     6.423305872147841e-13f,
	     2.382191739347918e-13f,

	     5.456116108943413e-12f,
	     4.878985199565852e-12f,
	     4.240448995017367e-12f,
	     3.559909094758253e-12f,
	     2.858043359288075e-12f,
	     2.156177623817898e-12f,
	     1.475637723558782e-12f,
	     8.371015190102975e-13f,
	     2.599706096327376e-13f,

	     -5.461314069809755e-12f,
	     -4.921085770524055e-12f,
	     -4.343405037091838e-12f,
	     -3.732668368707687e-12f,
	     -3.093523840190885e-12f,
	     -2.430835727329465e-12f,
	     -1.734679010007751e-12f,
	     -9.748253656609281e-13f,
	     -2.797435120168326e-13f,

	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     -2.283748241799531e-13f,
	     -4.037858874020686e-13f,
	     -2.146547464825323e-13f,
	     },
	    {
	     1.316524975873958e-01f, /* win[SHORT_TYPE] */
	     4.142135623730950e-01f,
	     7.673269879789602e-01f,

	     1.091308501069271e+00f, /* tantab_l */
	     1.303225372841206e+00f,
	     1.569685577117490e+00f,
	     1.920982126971166e+00f,
	     2.414213562373094e+00f,
	     3.171594802363212e+00f,
	     4.510708503662055e+00f,
	     7.595754112725146e+00f,
	     2.290376554843115e+01f,

	     0.98480775301220802032f, /* cx */
	     0.64278760968653936292f,
	     0.34202014332566882393f,
	     0.93969262078590842791f,
	     -0.17364817766693030343f,
	     -0.76604444311897790243f,
	     0.86602540378443870761f,
	     0.500000000000000e+00f,

	     -5.144957554275265e-01f, /* ca */
	     -4.717319685649723e-01f,
	     -3.133774542039019e-01f,
	     -1.819131996109812e-01f,
	     -9.457419252642064e-02f,
	     -4.096558288530405e-02f,
	     -1.419856857247115e-02f,
	     -3.699974673760037e-03f,

	     8.574929257125442e-01f, /* cs */
	     8.817419973177052e-01f,
	     9.496286491027329e-01f,
	     9.833145924917901e-01f,
	     9.955178160675857e-01f,
	     9.991605581781475e-01f,
	     9.998991952444470e-01f,
	     9.999931550702802e-01f,
	     },
	    {
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     0.000000000000000e+00f,
	     2.283748241799531e-13f,
	     4.037858874020686e-13f,
	     2.146547464825323e-13f,

	     5.461314069809755e-12f,
	     4.921085770524055e-12f,
	     4.343405037091838e-12f,
	     3.732668368707687e-12f,
	     3.093523840190885e-12f,
	     2.430835727329466e-12f,
	     1.734679010007751e-12f,
	     9.748253656609281e-13f,
	     2.797435120168326e-13f,

	     -5.456116108943413e-12f,
	     -4.878985199565852e-12f,
	     -4.240448995017367e-12f,
	     -3.559909094758253e-12f,
	     -2.858043359288075e-12f,
	     -2.156177623817898e-12f,
	     -1.475637723558782e-12f,
	     -8.371015190102975e-13f,
	     -2.599706096327376e-13f,

	     -2.382191739347913e-13f,
	     -6.423305872147834e-13f,
	     -9.400849094049688e-13f,
	     -1.122435026096556e-12f,
	     -1.183840321267481e-12f,
	     -1.122435026096556e-12f,
	     -9.400849094049688e-13f,
	     -6.423305872147841e-13f,
	     -2.382191739347918e-13f,
	     }
	};

	private static final float tantab_l[] = win[Encoder.SHORT_TYPE];
	private static final float[] cx = win[Encoder.SHORT_TYPE];
	private static final float[] ca = win[Encoder.SHORT_TYPE];
	private static final float[] cs = win[Encoder.SHORT_TYPE];

	/**
	 * new IDCT routine written by Takehiro TOMINAGA
	 * 
	 * PURPOSE: Overlapping window on PCM samples<BR>
	 * 
	 * SEMANTICS:<BR>
	 * 32 16-bit pcm samples are scaled to fractional 2's complement and
	 * concatenated to the end of the window buffer #x#. The updated window
	 * buffer #x# is then windowed by the analysis window #c# to produce the
	 * windowed sample #z#
	 */
	private static final int order[] = {
	    0, 1, 16, 17, 8, 9, 24, 25, 4, 5, 20, 21, 12, 13, 28, 29,
	    2, 3, 18, 19, 10, 11, 26, 27, 6, 7, 22, 23, 14, 15, 30, 31
	};

	/**
	 * returns sum_j=0^31 a[j]*cos(PI*j*(k+1/2)/32), 0<=k<32
	 */
	private void window_subband(final float[] x1, int x1Pos, final float a[]) {
		int wp = 10;

		int x2 = x1Pos + 238 - 14 - 286;

		for (int i = -15; i < 0; i++) {
			float w, s, t;

			w = enwindow[wp + -10];
			s = x1[x2 + -224] * w;
			t = x1[x1Pos + 224] * w;
			w = enwindow[wp + -9];
			s += x1[x2 + -160] * w;
			t += x1[x1Pos + 160] * w;
			w = enwindow[wp + -8];
			s += x1[x2 + -96] * w;
			t += x1[x1Pos + 96] * w;
			w = enwindow[wp + -7];
			s += x1[x2 + -32] * w;
			t += x1[x1Pos + 32] * w;
			w = enwindow[wp + -6];
			s += x1[x2 + 32] * w;
			t += x1[x1Pos + -32] * w;
			w = enwindow[wp + -5];
			s += x1[x2 + 96] * w;
			t += x1[x1Pos + -96] * w;
			w = enwindow[wp + -4];
			s += x1[x2 + 160] * w;
			t += x1[x1Pos + -160] * w;
			w = enwindow[wp + -3];
			s += x1[x2 + 224] * w;
			t += x1[x1Pos + -224] * w;

			w = enwindow[wp + -2];
			s += x1[x1Pos + -256] * w;
			t -= x1[x2 + 256] * w;
			w = enwindow[wp + -1];
			s += x1[x1Pos + -192] * w;
			t -= x1[x2 + 192] * w;
			w = enwindow[wp + 0];
			s += x1[x1Pos + -128] * w;
			t -= x1[x2 + 128] * w;
			w = enwindow[wp + 1];
			s += x1[x1Pos + -64] * w;
			t -= x1[x2 + 64] * w;
			w = enwindow[wp + 2];
			s += x1[x1Pos + 0] * w;
			t -= x1[x2 + 0] * w;
			w = enwindow[wp + 3];
			s += x1[x1Pos + 64] * w;
			t -= x1[x2 + -64] * w;
			w = enwindow[wp + 4];
			s += x1[x1Pos + 128] * w;
			t -= x1[x2 + -128] * w;
			w = enwindow[wp + 5];
			s += x1[x1Pos + 192] * w;
			t -= x1[x2 + -192] * w;

			/*
			 * this multiplyer could be removed, but it needs more 256 FLOAT
			 * data. thinking about the data cache performance, I think we
			 * should not use such a huge table. tt 2000/Oct/25
			 */
			s *= enwindow[wp + 6];
			w = t - s;
			a[30 + i * 2] = t + s;
			a[31 + i * 2] = enwindow[wp + 7] * w;
			wp += 18;
			x1Pos--;
			x2++;
		}
		{
			float s, t, u, v;
			t = x1[x1Pos + -16] * enwindow[wp + -10];
			s = x1[x1Pos + -32] * enwindow[wp + -2];
			t += (x1[x1Pos + -48] - x1[x1Pos + 16]) * enwindow[wp + -9];
			s += x1[x1Pos + -96] * enwindow[wp + -1];
			t += (x1[x1Pos + -80] + x1[x1Pos + 48]) * enwindow[wp + -8];
			s += x1[x1Pos + -160] * enwindow[wp + 0];
			t += (x1[x1Pos + -112] - x1[x1Pos + 80]) * enwindow[wp + -7];
			s += x1[x1Pos + -224] * enwindow[wp + 1];
			t += (x1[x1Pos + -144] + x1[x1Pos + 112]) * enwindow[wp + -6];
			s -= x1[x1Pos + 32] * enwindow[wp + 2];
			t += (x1[x1Pos + -176] - x1[x1Pos + 144]) * enwindow[wp + -5];
			s -= x1[x1Pos + 96] * enwindow[wp + 3];
			t += (x1[x1Pos + -208] + x1[x1Pos + 176]) * enwindow[wp + -4];
			s -= x1[x1Pos + 160] * enwindow[wp + 4];
			t += (x1[x1Pos + -240] - x1[x1Pos + 208]) * enwindow[wp + -3];
			s -= x1[x1Pos + 224];

			u = s - t;
			v = s + t;

			t = a[14];
			s = a[15] - t;

			a[31] = v + t; /* A0 */
			a[30] = u + s; /* A1 */
			a[15] = u - s; /* A2 */
			a[14] = v - t; /* A3 */
		}
		{
			float xr;
			xr = a[28] - a[0];
			a[0] += a[28];
			a[28] = xr * enwindow[wp + -2 * 18 + 7];
			xr = a[29] - a[1];
			a[1] += a[29];
			a[29] = xr * enwindow[wp + -2 * 18 + 7];

			xr = a[26] - a[2];
			a[2] += a[26];
			a[26] = xr * enwindow[wp + -4 * 18 + 7];
			xr = a[27] - a[3];
			a[3] += a[27];
			a[27] = xr * enwindow[wp + -4 * 18 + 7];

			xr = a[24] - a[4];
			a[4] += a[24];
			a[24] = xr * enwindow[wp + -6 * 18 + 7];
			xr = a[25] - a[5];
			a[5] += a[25];
			a[25] = xr * enwindow[wp + -6 * 18 + 7];

			xr = a[22] - a[6];
			a[6] += a[22];
			a[22] = xr * Util.SQRT2;
			xr = a[23] - a[7];
			a[7] += a[23];
			a[23] = xr * Util.SQRT2 - a[7];
			a[7] -= a[6];
			a[22] -= a[7];
			a[23] -= a[22];

			xr = a[6];
			a[6] = a[31] - xr;
			a[31] = a[31] + xr;
			xr = a[7];
			a[7] = a[30] - xr;
			a[30] = a[30] + xr;
			xr = a[22];
			a[22] = a[15] - xr;
			a[15] = a[15] + xr;
			xr = a[23];
			a[23] = a[14] - xr;
			a[14] = a[14] + xr;

			xr = a[20] - a[8];
			a[8] += a[20];
			a[20] = xr * enwindow[wp + -10 * 18 + 7];
			xr = a[21] - a[9];
			a[9] += a[21];
			a[21] = xr * enwindow[wp + -10 * 18 + 7];

			xr = a[18] - a[10];
			a[10] += a[18];
			a[18] = xr * enwindow[wp + -12 * 18 + 7];
			xr = a[19] - a[11];
			a[11] += a[19];
			a[19] = xr * enwindow[wp + -12 * 18 + 7];

			xr = a[16] - a[12];
			a[12] += a[16];
			a[16] = xr * enwindow[wp + -14 * 18 + 7];
			xr = a[17] - a[13];
			a[13] += a[17];
			a[17] = xr * enwindow[wp + -14 * 18 + 7];

			xr = -a[20] + a[24];
			a[20] += a[24];
			a[24] = xr * enwindow[wp + -12 * 18 + 7];
			xr = -a[21] + a[25];
			a[21] += a[25];
			a[25] = xr * enwindow[wp + -12 * 18 + 7];

			xr = a[4] - a[8];
			a[4] += a[8];
			a[8] = xr * enwindow[wp + -12 * 18 + 7];
			xr = a[5] - a[9];
			a[5] += a[9];
			a[9] = xr * enwindow[wp + -12 * 18 + 7];

			xr = a[0] - a[12];
			a[0] += a[12];
			a[12] = xr * enwindow[wp + -4 * 18 + 7];
			xr = a[1] - a[13];
			a[1] += a[13];
			a[13] = xr * enwindow[wp + -4 * 18 + 7];
			xr = a[16] - a[28];
			a[16] += a[28];
			a[28] = xr * enwindow[wp + -4 * 18 + 7];
			xr = -a[17] + a[29];
			a[17] += a[29];
			a[29] = xr * enwindow[wp + -4 * 18 + 7];

			xr = Util.SQRT2 * (a[2] - a[10]);
			a[2] += a[10];
			a[10] = xr;
			xr = Util.SQRT2 * (a[3] - a[11]);
			a[3] += a[11];
			a[11] = xr;
			xr = Util.SQRT2 * (-a[18] + a[26]);
			a[18] += a[26];
			a[26] = xr - a[18];
			xr = Util.SQRT2 * (-a[19] + a[27]);
			a[19] += a[27];
			a[27] = xr - a[19];

			xr = a[2];
			a[19] -= a[3];
			a[3] -= xr;
			a[2] = a[31] - xr;
			a[31] += xr;
			xr = a[3];
			a[11] -= a[19];
			a[18] -= xr;
			a[3] = a[30] - xr;
			a[30] += xr;
			xr = a[18];
			a[27] -= a[11];
			a[19] -= xr;
			a[18] = a[15] - xr;
			a[15] += xr;

			xr = a[19];
			a[10] -= xr;
			a[19] = a[14] - xr;
			a[14] += xr;
			xr = a[10];
			a[11] -= xr;
			a[10] = a[23] - xr;
			a[23] += xr;
			xr = a[11];
			a[26] -= xr;
			a[11] = a[22] - xr;
			a[22] += xr;
			xr = a[26];
			a[27] -= xr;
			a[26] = a[7] - xr;
			a[7] += xr;

			xr = a[27];
			a[27] = a[6] - xr;
			a[6] += xr;

			xr = Util.SQRT2 * (a[0] - a[4]);
			a[0] += a[4];
			a[4] = xr;
			xr = Util.SQRT2 * (a[1] - a[5]);
			a[1] += a[5];
			a[5] = xr;
			xr = Util.SQRT2 * (a[16] - a[20]);
			a[16] += a[20];
			a[20] = xr;
			xr = Util.SQRT2 * (a[17] - a[21]);
			a[17] += a[21];
			a[21] = xr;

			xr = -Util.SQRT2 * (a[8] - a[12]);
			a[8] += a[12];
			a[12] = xr - a[8];
			xr = -Util.SQRT2 * (a[9] - a[13]);
			a[9] += a[13];
			a[13] = xr - a[9];
			xr = -Util.SQRT2 * (a[25] - a[29]);
			a[25] += a[29];
			a[29] = xr - a[25];
			xr = -Util.SQRT2 * (a[24] + a[28]);
			a[24] -= a[28];
			a[28] = xr - a[24];

			xr = a[24] - a[16];
			a[24] = xr;
			xr = a[20] - xr;
			a[20] = xr;
			xr = a[28] - xr;
			a[28] = xr;

			xr = a[25] - a[17];
			a[25] = xr;
			xr = a[21] - xr;
			a[21] = xr;
			xr = a[29] - xr;
			a[29] = xr;

			xr = a[17] - a[1];
			a[17] = xr;
			xr = a[9] - xr;
			a[9] = xr;
			xr = a[25] - xr;
			a[25] = xr;
			xr = a[5] - xr;
			a[5] = xr;
			xr = a[21] - xr;
			a[21] = xr;
			xr = a[13] - xr;
			a[13] = xr;
			xr = a[29] - xr;
			a[29] = xr;

			xr = a[1] - a[0];
			a[1] = xr;
			xr = a[16] - xr;
			a[16] = xr;
			xr = a[17] - xr;
			a[17] = xr;
			xr = a[8] - xr;
			a[8] = xr;
			xr = a[9] - xr;
			a[9] = xr;
			xr = a[24] - xr;
			a[24] = xr;
			xr = a[25] - xr;
			a[25] = xr;
			xr = a[4] - xr;
			a[4] = xr;
			xr = a[5] - xr;
			a[5] = xr;
			xr = a[20] - xr;
			a[20] = xr;
			xr = a[21] - xr;
			a[21] = xr;
			xr = a[12] - xr;
			a[12] = xr;
			xr = a[13] - xr;
			a[13] = xr;
			xr = a[28] - xr;
			a[28] = xr;
			xr = a[29] - xr;
			a[29] = xr;

			xr = a[0];
			a[0] += a[31];
			a[31] -= xr;
			xr = a[1];
			a[1] += a[30];
			a[30] -= xr;
			xr = a[16];
			a[16] += a[15];
			a[15] -= xr;
			xr = a[17];
			a[17] += a[14];
			a[14] -= xr;
			xr = a[8];
			a[8] += a[23];
			a[23] -= xr;
			xr = a[9];
			a[9] += a[22];
			a[22] -= xr;
			xr = a[24];
			a[24] += a[7];
			a[7] -= xr;
			xr = a[25];
			a[25] += a[6];
			a[6] -= xr;
			xr = a[4];
			a[4] += a[27];
			a[27] -= xr;
			xr = a[5];
			a[5] += a[26];
			a[26] -= xr;
			xr = a[20];
			a[20] += a[11];
			a[11] -= xr;
			xr = a[21];
			a[21] += a[10];
			a[10] -= xr;
			xr = a[12];
			a[12] += a[19];
			a[19] -= xr;
			xr = a[13];
			a[13] += a[18];
			a[18] -= xr;
			xr = a[28];
			a[28] += a[3];
			a[3] -= xr;
			xr = a[29];
			a[29] += a[2];
			a[2] -= xr;
		}
	}

	/**
	 * Function: Calculation of the MDCT In the case of long blocks (type 0,1,3)
	 * there are 36 coefficents in the time domain and 18 in the frequency
	 * domain.<BR>
	 * In the case of short blocks (type 2) there are 3 transformations with
	 * short length. This leads to 12 coefficents in the time and 6 in the
	 * frequency domain. In this case the results are stored side by side in the
	 * vector out[].
	 * 
	 * New layer3
	 */
	private void mdct_short(final float[] inout, int inoutPos) {
		for (int l = 0; l < 3; l++) {
			float tc0, tc1, tc2, ts0, ts1, ts2;

			ts0 = inout[inoutPos + 2 * 3] * win[Encoder.SHORT_TYPE][0]
					- inout[inoutPos + 5 * 3];
			tc0 = inout[inoutPos + 0 * 3] * win[Encoder.SHORT_TYPE][2]
					- inout[inoutPos + 3 * 3];
			tc1 = ts0 + tc0;
			tc2 = ts0 - tc0;

			ts0 = inout[inoutPos + 5 * 3] * win[Encoder.SHORT_TYPE][0]
					+ inout[inoutPos + 2 * 3];
			tc0 = inout[inoutPos + 3 * 3] * win[Encoder.SHORT_TYPE][2]
					+ inout[inoutPos + 0 * 3];
			ts1 = ts0 + tc0;
			ts2 = -ts0 + tc0;

			tc0 = (inout[inoutPos + 1 * 3] * win[Encoder.SHORT_TYPE][1] - inout[inoutPos + 4 * 3]) * 2.069978111953089e-11f;
			/*
			 * tritab_s [ 1 ]
			 */
			ts0 = (inout[inoutPos + 4 * 3] * win[Encoder.SHORT_TYPE][1] + inout[inoutPos + 1 * 3]) * 2.069978111953089e-11f;
			/*
			 * tritab_s [ 1 ]
			 */
			inout[inoutPos + 3 * 0] = tc1 * 1.907525191737280e-11f + tc0;
			/*
			 * tritab_s[ 2 ]
			 */
			inout[inoutPos + 3 * 5] = -ts1 * 1.907525191737280e-11f + ts0;
			/*
			 * tritab_s[0 ]
			 */
			tc2 = tc2 * 0.86602540378443870761f * 1.907525191737281e-11f;
			/*
			 * tritab_s[ 2]
			 */
			ts1 = ts1 * 0.5f * 1.907525191737281e-11f + ts0;
			inout[inoutPos + 3 * 1] = tc2 - ts1;
			inout[inoutPos + 3 * 2] = tc2 + ts1;

			tc1 = tc1 * 0.5f * 1.907525191737281e-11f - tc0;
			ts2 = ts2 * 0.86602540378443870761f * 1.907525191737281e-11f;
			/*
			 * tritab_s[ 0]
			 */
			inout[inoutPos + 3 * 3] = tc1 + ts2;
			inout[inoutPos + 3 * 4] = tc1 - ts2;

			inoutPos++;
		}
	}

	final void mdct_long(final float[] out, final int outPos, final float[] in) {
		float ct, st;
		{
			float tc1, tc2, tc3, tc4, ts5, ts6, ts7, ts8;
			/* 1,2, 5,6, 9,10, 13,14, 17 */
			tc1 = in[17] - in[9];
			tc3 = in[15] - in[11];
			tc4 = in[14] - in[12];
			ts5 = in[0] + in[8];
			ts6 = in[1] + in[7];
			ts7 = in[2] + in[6];
			ts8 = in[3] + in[5];

			out[outPos + 17] = (ts5 + ts7 - ts8) - (ts6 - in[4]);
			st = (ts5 + ts7 - ts8) * cx[12 + 7] + (ts6 - in[4]);
			ct = (tc1 - tc3 - tc4) * cx[12 + 6];
			out[outPos + 5] = ct + st;
			out[outPos + 6] = ct - st;

			tc2 = (in[16] - in[10]) * cx[12 + 6];
			ts6 = ts6 * cx[12 + 7] + in[4];
			ct = tc1 * cx[12 + 0] + tc2 + tc3 * cx[12 + 1] + tc4 * cx[12 + 2];
			st = -ts5 * cx[12 + 4] + ts6 - ts7 * cx[12 + 5] + ts8 * cx[12 + 3];
			out[outPos + 1] = ct + st;
			out[outPos + 2] = ct - st;

			ct = tc1 * cx[12 + 1] - tc2 - tc3 * cx[12 + 2] + tc4 * cx[12 + 0];
			st = -ts5 * cx[12 + 5] + ts6 - ts7 * cx[12 + 3] + ts8 * cx[12 + 4];
			out[outPos + 9] = ct + st;
			out[outPos + 10] = ct - st;

			ct = tc1 * cx[12 + 2] - tc2 + tc3 * cx[12 + 0] - tc4 * cx[12 + 1];
			st = ts5 * cx[12 + 3] - ts6 + ts7 * cx[12 + 4] - ts8 * cx[12 + 5];
			out[outPos + 13] = ct + st;
			out[outPos + 14] = ct - st;
		}
		{
			float ts1, ts2, ts3, ts4, tc5, tc6, tc7, tc8;

			ts1 = in[8] - in[0];
			ts3 = in[6] - in[2];
			ts4 = in[5] - in[3];
			tc5 = in[17] + in[9];
			tc6 = in[16] + in[10];
			tc7 = in[15] + in[11];
			tc8 = in[14] + in[12];

			out[outPos + 0] = (tc5 + tc7 + tc8) + (tc6 + in[13]);
			ct = (tc5 + tc7 + tc8) * cx[12 + 7] - (tc6 + in[13]);
			st = (ts1 - ts3 + ts4) * cx[12 + 6];
			out[outPos + 11] = ct + st;
			out[outPos + 12] = ct - st;

			ts2 = (in[7] - in[1]) * cx[12 + 6];
			tc6 = in[13] - tc6 * cx[12 + 7];
			ct = tc5 * cx[12 + 3] - tc6 + tc7 * cx[12 + 4] + tc8 * cx[12 + 5];
			st = ts1 * cx[12 + 2] + ts2 + ts3 * cx[12 + 0] + ts4 * cx[12 + 1];
			out[outPos + 3] = ct + st;
			out[outPos + 4] = ct - st;

			ct = -tc5 * cx[12 + 5] + tc6 - tc7 * cx[12 + 3] - tc8 * cx[12 + 4];
			st = ts1 * cx[12 + 1] + ts2 - ts3 * cx[12 + 2] - ts4 * cx[12 + 0];
			out[outPos + 7] = ct + st;
			out[outPos + 8] = ct - st;

			ct = -tc5 * cx[12 + 4] + tc6 - tc7 * cx[12 + 5] - tc8 * cx[12 + 3];
			st = ts1 * cx[12 + 0] - ts2 + ts3 * cx[12 + 1] - ts4 * cx[12 + 2];
			out[outPos + 15] = ct + st;
			out[outPos + 16] = ct - st;
		}
	}

	public final void mdct_sub48(final LameInternalFlags gfc, final float[] w0,
			final float[] w1) {
		float[] wk = w0;
		int wkPos = 286;
		/* thinking cache performance, ch->gr loop is better than gr->ch loop */
		for (int ch = 0; ch < gfc.channels_out; ch++) {
			for (int gr = 0; gr < gfc.mode_gr; gr++) {
				int band;
				final GrInfo gi = (gfc.l3_side.tt[gr][ch]);
				float[] mdct_enc = gi.xr;
				int mdct_encPos = 0;
				float[][] samp = gfc.sb_sample[ch][1 - gr];
				int sampPos = 0;

				for (int k = 0; k < 18 / 2; k++) {
					window_subband(wk, wkPos, samp[sampPos]);
					window_subband(wk, wkPos + 32, samp[sampPos + 1]);
					sampPos += 2;
					wkPos += 64;
					/*
					 * Compensate for inversion in the analysis filter
					 */
					for (band = 1; band < 32; band += 2) {
						samp[sampPos - 1][band] *= -1;
					}
				}

				/*
				 * Perform imdct of 18 previous subband samples + 18 current
				 * subband samples
				 */
				for (band = 0; band < 32; band++, mdct_encPos += 18) {
					int type = gi.block_type;
					float[][] band0 = gfc.sb_sample[ch][gr];
					float[][] band1 = gfc.sb_sample[ch][1 - gr];
					if (gi.mixed_block_flag != 0 && band < 2)
						type = 0;
					if (gfc.amp_filter[band] < 1e-12) {
						Arrays.fill(mdct_enc, mdct_encPos + 0,
								mdct_encPos + 18, 0);
					} else {
						if (gfc.amp_filter[band] < 1.0) {
							for (int k = 0; k < 18; k++)
								band1[k][order[band]] *= gfc.amp_filter[band];
						}
						if (type == Encoder.SHORT_TYPE) {
							for (int k = -NS / 4; k < 0; k++) {
								float w = win[Encoder.SHORT_TYPE][k + 3];
								mdct_enc[mdct_encPos + k * 3 + 9] = band0[9 + k][order[band]]
										* w - band0[8 - k][order[band]];
								mdct_enc[mdct_encPos + k * 3 + 18] = band0[14 - k][order[band]]
										* w + band0[15 + k][order[band]];
								mdct_enc[mdct_encPos + k * 3 + 10] = band0[15 + k][order[band]]
										* w - band0[14 - k][order[band]];
								mdct_enc[mdct_encPos + k * 3 + 19] = band1[2 - k][order[band]]
										* w + band1[3 + k][order[band]];
								mdct_enc[mdct_encPos + k * 3 + 11] = band1[3 + k][order[band]]
										* w - band1[2 - k][order[band]];
								mdct_enc[mdct_encPos + k * 3 + 20] = band1[8 - k][order[band]]
										* w + band1[9 + k][order[band]];
							}
							mdct_short(mdct_enc, mdct_encPos);
						} else {
							float work[] = new float[18];
							for (int k = -NL / 4; k < 0; k++) {
								float a, b;
								a = win[type][k + 27]
										* band1[k + 9][order[band]]
										+ win[type][k + 36]
										* band1[8 - k][order[band]];
								b = win[type][k + 9]
										* band0[k + 9][order[band]]
										- win[type][k + 18]
										* band0[8 - k][order[band]];
								work[k + 9] = a - b * tantab_l[3 + k + 9];
								work[k + 18] = a * tantab_l[3 + k + 9] + b;
							}

							mdct_long(mdct_enc, mdct_encPos, work);
						}
					}
					/*
					 * Perform aliasing reduction butterfly
					 */
					if (type != Encoder.SHORT_TYPE && band != 0) {
						for (int k = 7; k >= 0; --k) {
							float bu, bd;
							bu = mdct_enc[mdct_encPos + k] * ca[20 + k]
									+ mdct_enc[mdct_encPos + -1 - k]
									* cs[28 + k];
							bd = mdct_enc[mdct_encPos + k] * cs[28 + k]
									- mdct_enc[mdct_encPos + -1 - k]
									* ca[20 + k];

							mdct_enc[mdct_encPos + -1 - k] = bu;
							mdct_enc[mdct_encPos + k] = bd;
						}
					}
				}
			}
			wk = w1;
			wkPos = 286;
			if (gfc.mode_gr == 1) {
				for (int i = 0; i < 18; i++) {
					System.arraycopy(gfc.sb_sample[ch][1][i], 0,
							gfc.sb_sample[ch][0][i], 0, 32);
				}
			}
		}
	}
}
